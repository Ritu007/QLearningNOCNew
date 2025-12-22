import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import random
import uuid
from collections import defaultdict
from environment import Node, Link  # your existing environment

style.use("ggplot")

# ---------------------------
# Basic grid / training params
# ---------------------------
SIZE = 5                    # grid half-extent used earlier: grid coords range from -SIZE+1 .. SIZE-1
HM_EPISODES = 2000          # reduce for testing; set to 10001 if you want original runs
MOVE_PENALTY = 10
OBSTACLE = 200
DEST_REWARD = 50

# exploration
EPSILON = 0.95
EPS_DECAY = 0.999

SHOW_EVERY = 200
SHOW_LAST = 50
EXPLORATION_STEPS = 500

LEARNING_RATE = 0.1
DISCOUNT = 0.95

# ---------------------------
# Fault / multi-packet params
# ---------------------------
MAX_NODE_FAULTS = 4
MAX_LINK_FAULTS = 0
FAULT_MOVE_PENALTY = 50
MAX_ACTIVE_PACKETS = 6       # concurrent packets
PACKET_INJECTION_PROB = 0.25
PACKET_MAX_AGE = 200
MAX_TIMESTEPS_PER_EPISODE = 400

# ---------------------------
# Reward component weights
# ---------------------------
W_MOVE = 1.0
W_DIRECTION = 1.0
W_FAULT = 1.0
W_CONGESTION = 1.0

BLOCKED_INCREASES_CONGESTION = True

# congestion controls
RESET_CONGESTION_EACH_EPISODE = True   # set True to fully reset each episode (Quick Fix A)
CONGESTION_DECAY = 0.90                 # keep 0.0..1.0; lower => faster forgetting (Fix B)
USE_DECAY = True                        # set True to apply decay each episode
MAX_NODE_VISIT_CAP = 100                # cap contribution per node (Fix C)
MAX_LINK_VISIT_CAP = 100

# ---------------------------
# Utility / colors for visualization
# ---------------------------
SOURCE_N = 1
DEST_N = 2

COLOR_MAP = {
    1: (255, 175, 0),   # source color (not used for multi packets but kept)
    2: (0, 255, 0),     # dest color
}

# ---------------------------
# Q-table helpers
# ---------------------------
def init_q_table(start_q_table=None):
    """
    Initialize Q-table keys for all possible observation differences:
    observations are (dx, dy) where dx,dy in [-SIZE+1, SIZE-1]
    """
    if start_q_table is None:
        q_table = {}
        for x in range(-SIZE + 1, SIZE):
            for y in range(-SIZE + 1, SIZE):
                q_table[(x, y)] = [np.random.uniform(-5, 0) for _ in range(4)]
    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)
    return q_table

def clip_obs(obs):
    """
    Clip an obs (dx, dy) into the valid q_table key range.
    This ensures we DON'T expand q_table keys (state size remains same).
    """
    dx = int(np.clip(obs[0], -SIZE + 1, SIZE - 1))
    dy = int(np.clip(obs[1], -SIZE + 1, SIZE - 1))
    return (dx, dy)

# ---------------------------
# Grid nodes & links helpers
# ---------------------------
def all_grid_nodes():
    return [(x, y) for x in range(0, SIZE) for y in range(0, SIZE)]

def all_adjacent_links():
    nodes = all_grid_nodes()
    links = set()
    for (x, y) in nodes:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxn = (x + dx, y + dy)
            if nxn in nodes:
                links.add(frozenset({(x, y), nxn}))
    return list(links)

ALL_POSSIBLE_LINKS = all_adjacent_links()

def generate_random_faults(avoid_src, avoid_dst,
                           max_node_faults=MAX_NODE_FAULTS,
                           max_link_faults=MAX_LINK_FAULTS):
    nodes_space = [n for n in all_grid_nodes() if n != avoid_src and n != avoid_dst]
    node_fault_count = np.random.randint(0, max_node_faults + 1)
    faulty_nodes = set(random.sample(nodes_space, k=node_fault_count)) if node_fault_count > 0 else set()

    possible_links = [L for L in ALL_POSSIBLE_LINKS if (avoid_src not in L and avoid_dst not in L)]
    link_fault_count = np.random.randint(0, max_link_faults + 1)
    faulty_links = set(random.sample(possible_links, k=link_fault_count)) if link_fault_count > 0 else set()

    return faulty_nodes, faulty_links

# ---------------------------
# Reward computation
# ---------------------------
# ---------------------------
# Neighbourhood helpers
# ---------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def nodes_in_neighbourhood(center, radius):
    """Return list of nodes (x,y) with Manhattan distance <= radius"""
    cx, cy = center
    nodes = []
    for x in range(0, SIZE):
        for y in range(0, SIZE):
            if manhattan((cx, cy), (x, y)) <= radius:
                nodes.append((x, y))
    return nodes

def links_in_neighbourhood(center, radius):
    """Return list of frozenset links where at least one endpoint lies in the neighbourhood.
       You can tighten this to only include links whose mid-point is inside, but endpoint-based is sufficient."""
    neigh_nodes = set(nodes_in_neighbourhood(center, radius))
    links = []
    for link in ALL_POSSIBLE_LINKS:
        # link is frozenset({n1, n2})
        if any(n in neigh_nodes for n in link):
            links.append(link)
    return links

# ---------------------------
# neighbourhood congestion calculator
# ---------------------------
def neighbourhood_congestion(center, radius, node_visit_count, link_visit_count, active_packets=None):
    """
    Returns a normalized congestion score in [0, 1] (higher => more congested).
    - center: (x,y)
    - radius: Manhattan radius
    - node_visit_count, link_visit_count: dict-like counters
    - active_packets: list of Packet instances (optional) to estimate immediate occupancy
    """
    neigh_nodes = nodes_in_neighbourhood(center, radius)
    neigh_links = links_in_neighbourhood(center, radius)

    # Sum historical visit counts (could be big -> normalize by area * scale)
    total_node_visits = sum(node_visit_count.get(n, 0) for n in neigh_nodes)
    total_link_visits = sum(link_visit_count.get(l, 0) for l in neigh_links)

    # Immediate active-packet count in neighbourhood
    active_in_neigh = 0
    if active_packets is not None:
        for pkt in active_packets:
            if manhattan(pkt.pos, center) <= radius:
                active_in_neigh += 1

    # Normalization factors: area sizes. Use max expected counts to avoid runaway values.
    node_area = max(1, len(neigh_nodes))
    link_area = max(1, len(neigh_links))

    # scale hist_visits into a 0..1 range using a soft cap (tunable)
    # HISTORY_SCALE = expected visits per node for high congestion (tweak)
    HISTORY_SCALE = 10.0  # if average node_visit_count ~10 => considered high
    hist_score = (total_node_visits / node_area + total_link_visits / link_area) / HISTORY_SCALE
    hist_score = float(np.tanh(hist_score))  # squashing to 0..<1 (smooth)

    # active packets scale: if many active packets in neighbourhood -> high immediate congestion
    # ACTIVE_SCALE = expected active packets for high congestion in that radius
    ACTIVE_SCALE = max(1.0, node_area * 0.5)  # e.g., 50% of nodes filled is high
    active_score = active_in_neigh / ACTIVE_SCALE
    active_score = float(np.tanh(active_score))

    # combine (weights tuneable)
    W_HIST = 0.4
    W_ACTIVE = 0.6
    combined = W_HIST * hist_score + W_ACTIVE * active_score

    # ensure 0..1
    combined = float(np.clip(combined, 0.0, 1.0))
    return combined, {
        "hist_score": hist_score,
        "active_score": active_score,
        "active_in_neigh": active_in_neigh,
        "neigh_nodes": len(neigh_nodes),
        "neigh_links": len(neigh_links)
    }

# ---------------------------
# Updated compute_reward_v2 with neighbourhood congestion
# ---------------------------
# Add two new tunable hyperparams:
NEIGH_RADIUS = 1          # 1 or 2; radius around the packet to check congestion
W_NEIGH_CONGESTION = 30.0  # how strongly to penalize neighbourhood congestion (tunable)

def compute_reward(prev_node, new_node, dest,
                      traversed_link,
                      was_blocked,
                      faulty_nodes, faulty_links,
                      node_visit_count, link_visit_count,
                      active_packets=None,
                      neighbourhood_radius=NEIGH_RADIUS):
    """
    Builds on compute_reward: adds neighbourhood congestion penalty computed by neighbourhood_congestion(...)
    Returns (total_reward, details)
    """
    details = {}

    # base components (same as previous compute_reward)
    if new_node == prev_node:
        base_move = MOVE_PENALTY + 10
    else:
        base_move = MOVE_PENALTY
    details['base_move'] = -W_MOVE * base_move

    prev_dist = manhattan(prev_node, dest)
    new_dist = manhattan(new_node, dest)
    dist_progress = prev_dist - new_dist
    max_dist = 2 * (SIZE - 1) if SIZE > 1 else 1
    norm_progress = dist_progress / max_dist
    direction_component = W_DIRECTION * norm_progress
    details['direction'] = direction_component

    fault_component = 0.0
    if was_blocked:
        fault_component = -W_FAULT
    else:
        if new_node in faulty_nodes:
            fault_component = -W_FAULT
        elif traversed_link is not None and traversed_link in faulty_links:
            fault_component = -W_FAULT
    details['fault'] = fault_component

    node_count = node_visit_count.get(new_node, 0)
    link_count = link_visit_count.get(traversed_link, 0) if traversed_link is not None else 0
    congestion_component = -W_CONGESTION * (node_count + link_count)
    details['local_congestion'] = congestion_component

    # ----------------------
    # neighbourhood congestion component (NEW)
    # ----------------------
    neigh_score, neigh_info = neighbourhood_congestion(center=new_node,
                                                      radius=neighbourhood_radius,
                                                      node_visit_count=node_visit_count,
                                                      link_visit_count=link_visit_count,
                                                      active_packets=active_packets)
    # convert neigh_score (0..1) into reward penalty (negative)
    neigh_component = - W_NEIGH_CONGESTION * neigh_score
    details['neighbourhood_score'] = neigh_score
    details['neighbourhood_penalty'] = neigh_component
    details.update(neigh_info)  # include hist_score, active_score, etc.

    # dest reward
    if new_node == dest:
        dest_comp = DEST_REWARD
    else:
        dest_comp = 0
    details['dest'] = dest_comp

    total_reward = details['base_move'] + details['direction'] + details['fault'] + details['local_congestion'] + details['neighbourhood_penalty'] + details['dest']
    details['total'] = total_reward
    details['prev_dist'] = prev_dist
    details['new_dist'] = new_dist
    details['norm_progress'] = norm_progress

    return total_reward, details


def compute_reward_v1(prev_node, new_node, dest,
                   traversed_link,
                   was_blocked,
                   faulty_nodes, faulty_links,
                   node_visit_count, link_visit_count):
    """
    Multi-part dynamic reward:
     - move penalty (base)
     - directional progress (closer/farther from destination)
     - fault penalty (blocked or traversed faulty)
     - congestion penalty (based on visit counts)
     - destination reward
    """
    details = {}

    # 1) Base move penalty
    if new_node == prev_node:
        base_move = MOVE_PENALTY + 10
    else:
        base_move = MOVE_PENALTY
    details['base_move'] = -W_MOVE * base_move

    # 2) Direction/progress
    prev_dist = manhattan(prev_node, dest)
    new_dist = manhattan(new_node, dest)
    dist_progress = prev_dist - new_dist  # +ve if moved closer
    max_dist = 2 * (SIZE - 1) if SIZE > 1 else 1
    norm_progress = dist_progress / max_dist
    direction_component = W_DIRECTION * norm_progress
    details['direction'] = direction_component

    # 3) Fault penalty
    fault_component = 0.0
    if was_blocked:
        fault_component = -W_FAULT
    else:
        if new_node in faulty_nodes:
            fault_component = -W_FAULT
        elif traversed_link is not None and traversed_link in faulty_links:
            fault_component = -W_FAULT
    details['fault'] = fault_component

    # 4) Congestion penalty
    node_count = node_visit_count.get(new_node, 0)
    link_count = link_visit_count.get(traversed_link, 0) if traversed_link is not None else 0
    congestion_component = -W_CONGESTION * (node_count + link_count)
    details['congestion'] = congestion_component

    # 5) Destination reward
    if new_node == dest:
        dest_comp = DEST_REWARD
    else:
        dest_comp = 0
    details['dest'] = dest_comp

    total_reward = details['base_move'] + details['direction'] + details['fault'] + details['congestion'] + details['dest']
    details['total'] = total_reward
    details['prev_dist'] = prev_dist
    details['new_dist'] = new_dist
    details['norm_progress'] = norm_progress

    return total_reward, details

# ---------------------------
# Packet class
# ---------------------------
class Packet:
    def __init__(self, src, dst):
        self.id = str(uuid.uuid4())[:8]
        self.src = src
        self.dst = dst
        # create a Node and set to src
        self.node = Node()
        self.node.x, self.node.y = src
        self.done = False
        self.age = 0
        self.cum_reward = 0.0

    @property
    def pos(self):
        return (self.node.x, self.node.y)

# ---------------------------
# Initialization
# ---------------------------
q_table = init_q_table()
print("Q table keys:", len(q_table))

episode_rewards = []
total_moves = []
expected_moves = []

# shared congestion counters (can be reset per-episode or persistent across episodes)
node_visit_count = defaultdict(int)
link_visit_count = defaultdict(int)

# ---------------------------
# TRAINING / EPISODE LOOP
# ---------------------------
for episode in range(HM_EPISODES):
    # representative pair to use for fault avoidance and q clipping

    if RESET_CONGESTION_EACH_EPISODE:
        node_visit_count.clear()
        link_visit_count.clear()

    rep_src = Node()
    rep_dst = Node()
    rep_source = (rep_src.x, rep_src.y)
    rep_dest = (rep_dst.x, rep_dst.y)

    # generate faults for this episode (avoid rep src/dst)
    faulty_nodes, faulty_links = generate_random_faults(rep_source, rep_dest)

    show = (episode % SHOW_EVERY == 0) or (episode >= HM_EPISODES - SHOW_LAST)
    if episode % SHOW_EVERY == 0:
        print(f"Episode {episode} | epsilon: {EPSILON:.4f} | last {SHOW_EVERY} mean reward: {np.mean(episode_rewards[-SHOW_EVERY:]) if episode_rewards else 0}")
        print("Faulty nodes:", faulty_nodes)
        print("Faulty links:", [tuple(list(x)) for x in faulty_links])

    # Optionally reset congestion counters each episode (comment to make congestion persistent)
    # node_visit_count.clear()
    # link_visit_count.clear()

    active_packets = []
    episode_reward = 0.0
    timestep = 0

    # ensure q_table keys exist for the representative dest (we do not add new keys, only ensure clipping works)
    # (init_q_table already populated keys for the full obs range so nothing to do here)

    while timestep < MAX_TIMESTEPS_PER_EPISODE:
        timestep += 1

        # Packet injection
        if len(active_packets) < MAX_ACTIVE_PACKETS and np.random.random() < PACKET_INJECTION_PROB:
            # choose random src and dst
            src = random.choice(all_grid_nodes())
            dst = random.choice(all_grid_nodes())
            while dst in faulty_nodes:
                dst = random.choice(all_grid_nodes())
            while src in faulty_nodes:
                src = random.choice(all_grid_nodes())
            while dst == src:
                dst = random.choice(all_grid_nodes())

            p = Packet(src=src, dst=dst)
            active_packets.append(p)
            if show:
                print(f"Injected pkt {p.id} {src} -> {dst}")

        if len(active_packets) == 0 and timestep > 10 and episode > EXPLORATION_STEPS:
            # optionally early-stop the episode if nothing to do
            break

        # service packets this timestep: resolve link contention via occupied_links set
        occupied_links = set()
        random.shuffle(active_packets)  # random service order to reduce deterministic bias

        for pkt in list(active_packets):  # iterate over a shallow copy
            if pkt.done:
                try:
                    active_packets.remove(pkt)
                except ValueError:
                    pass
                continue

            prev_node = pkt.pos
            obs_raw = (pkt.dst[0] - prev_node[0], pkt.dst[1] - prev_node[1])
            obs = clip_obs(obs_raw)

            # action selection using clipped obs (so q_table not expanded)
            if np.random.random() > EPSILON and episode > EXPLORATION_STEPS:
                action = int(np.argmax(q_table[obs]))
            else:
                action = int(np.random.randint(0, 4))

            # apply action on packet-local node
            pkt.node.action(action)
            new_node = pkt.pos
            traversed_link = frozenset({prev_node, new_node}) if prev_node != new_node else None

            # determine blockage: faults or contention
            was_blocked = False
            if new_node in faulty_nodes:
                was_blocked = True
            if traversed_link is not None and traversed_link in faulty_links:
                was_blocked = True
            if traversed_link is not None and traversed_link in occupied_links:
                was_blocked = True

            # revert if blocked
            if was_blocked:
                pkt.node.x, pkt.node.y = prev_node
                new_node = prev_node
                traversed_link = None
                if BLOCKED_INCREASES_CONGESTION:
                    node_visit_count[prev_node] += 1
            else:
                # successful traversal: mark occupancy and update congestion counters
                if traversed_link is not None:
                    occupied_links.add(traversed_link)
                node_visit_count[new_node] += 1
                if traversed_link is not None:
                    link_visit_count[traversed_link] += 1

            # compute reward for this step (packet-centric)
            # reward, reward_details = compute_reward(prev_node=prev_node,
            #                                         new_node=new_node,
            #                                         dest=pkt.dst,
            #                                         traversed_link=traversed_link,
            #                                         was_blocked=was_blocked,
            #                                         faulty_nodes=faulty_nodes,
            #                                         faulty_links=faulty_links,
            #                                         node_visit_count=node_visit_count,
            #                                         link_visit_count=link_visit_count)

            reward, reward_details = compute_reward(prev_node=prev_node,
                                                       new_node=new_node,
                                                       dest=pkt.dst,
                                                       traversed_link=traversed_link,
                                                       was_blocked=was_blocked,
                                                       faulty_nodes=faulty_nodes,
                                                       faulty_links=faulty_links,
                                                       node_visit_count=node_visit_count,
                                                       link_visit_count=link_visit_count,
                                                       active_packets=active_packets,  # pass current list
                                                       neighbourhood_radius=NEIGH_RADIUS)

            pkt.cum_reward += reward
            episode_reward += reward
            pkt.age += 1

            # Q-learning update: use clipped obs and clipped new_obs
            new_obs_raw = (pkt.dst[0] - new_node[0], pkt.dst[1] - new_node[1])
            new_obs = clip_obs(new_obs_raw)
            max_future_q = np.max(q_table[new_obs])
            current_q = q_table[obs][action]

            if new_node == pkt.dst:
                target_q = DEST_REWARD
                pkt.done = True
            else:
                target_q = reward + DISCOUNT * max_future_q

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * target_q
            q_table[obs][action] = new_q

            # delivered or aged out removal
            if pkt.done or pkt.age > PACKET_MAX_AGE:
                if pkt in active_packets:
                    active_packets.remove(pkt)
                if pkt.done and show:
                    print(f"Packet {pkt.id} delivered (age {pkt.age}) cum_reward {pkt.cum_reward:.2f}")

        # optional visualization: draw faulty nodes and active packets
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            # mark faulty nodes
            for fn in faulty_nodes:
                fx, fy = fn
                # env uses [row=y][col=x] indexing like your original code
                if 0 <= fy < SIZE*1 and 0 <= fx < SIZE*1:
                    # coordinates are already in range -SIZE+1..SIZE-1; we index by offsetting to 0..SIZE-1
                    pass
            # Because your original env indexing used directly end_node.y/end_node.x, here coordinates
            # must be mapped into array indexed 0..SIZE-1. We'll map (x,y) [-SIZE+1..SIZE-1] -> [0..SIZE-1]:
            def idx(coord):
                # coord is (x,y)
                x, y = coord
                return (y + SIZE - 1, x + SIZE - 1)  # row, col

            for fn in faulty_nodes:
                # r, c = idx(fn)
                r, c = fn
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    env[r][c] = (0, 0, 255)  # red for faulty nodes

            # mark active packets
            for pkt in active_packets:
                # r, c = idx(pkt.pos)
                r, c = pkt.pos
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    env[r][c] = COLOR_MAP[1]  # orange

            # optionally mark rep_dest
            # r, c = idx(rep_dest)
            r, c = rep_dest
            if 0 <= r < SIZE and 0 <= c < SIZE:
                env[r][c] = COLOR_MAP[2]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300), resample=Image.BOX)
            cv2.imshow("", np.array(img))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # epsilon decay per timestep after some exploration period
        if episode > EXPLORATION_STEPS:
            EPSILON *= EPS_DECAY

    # end episode bookkeeping
    episode_rewards.append(episode_reward)
    total_moves.append(timestep)
    expected_moves.append(0)

    if episode % SHOW_EVERY == 0 or episode >= HM_EPISODES - SHOW_LAST:
        print(f"Episode {episode} done | timesteps: {timestep} | total reward: {episode_reward:.2f}")

# ---------------------------
# Post-training visualizations & save q_table
# ---------------------------
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel("Cumulative Reward")
plt.xlabel("Episodes")
plt.show()

with open(f"q_table-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

print("Training finished. Q-table saved.")
