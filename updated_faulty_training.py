# noc_qlearning_4x4_absstate.py
import numpy as np
import random
import time
import pickle
from collections import defaultdict
from environment import Node    # assumes Node.action(action) updates node.x,node.y

# -------------------------
# Environment / grid config
# -------------------------
GRID_W = GRID_H = 5        # 5x5 grid
NUM_NODES = GRID_W * GRID_H  # 25
NUM_ACTIONS = 4              # UP, RIGHT, DOWN, LEFT

# state components sizes
NX = NUM_NODES               # x = current node index 0..15
NY = NUM_NODES               # y = destination node index 0..15
NC = 4                       # cong levels 0..3
ND = 4                       # dir values 0..3

NUM_DIR = 4 #RIGHT-DOWN, DOWN-LEFT, LEFT-UP, UP-RIGHT
NEIGHBOUR_SIZE = 8

# NUM_STATES = NX * NY * NC * ND

NUM_STATES = NUM_DIR * (2 ** NEIGHBOUR_SIZE)

# -------------------------
# Training hyperparams
# -------------------------
HM_EPISODES = 3000        # increase for serious training
MAX_TIMESTEPS = 400
MAX_STEPS_PER_EPISODE = 400

LEARNING_RATE = 0.1
DISCOUNT = 0.95

EPSILON = 0.95
EPS_DECAY = 0.9995

# multi-packet params
MAX_ACTIVE_PACKETS = 6
PACKET_INJECTION_PROB = 0.2
PACKET_MAX_AGE = 200

# -------------------------
# Reward hyperparams & caps
# -------------------------
MOVE_PENALTY = 10
DEST_REWARD = 50

W_MOVE = 1.0
W_DIR = 20.0
W_FAULT = 60.0
W_LOCAL_CONG = 5.0
W_NEIGH_CONG = 30.0

# caps and scales
HISTORY_SCALE = 8.0            # expected visit count considered "high"
MAX_NODE_VISIT_CAP = 50.0
MAX_LINK_VISIT_CAP = 50.0
R_MAX = 1000.0                 # clip per-step reward to [-R_MAX, R_MAX]

# congestion decay per episode (prevents runaway)
USE_DECAY = True
CONGESTION_DECAY = 0.92

# -------------------------
# Utility mappings
# -------------------------
PORTS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
DELTAS = {
    0: (0, -1),   # UP    (x, y) -> (x, y-1)
    1: (1, 0),    # RIGHT
    2: (0, 1),    # DOWN
    3: (-1, 0)    # LEFT
}

# -------------------------
# Q-table initialization & memory info
# -------------------------
q_table = np.random.uniform(-1, 0, size=(NUM_STATES, NUM_ACTIONS)).astype(np.float32)
print(f"Q-table shape: {q_table.shape}, bytes: {q_table.nbytes} (~{q_table.nbytes/1024:.1f} KB)")

# -------------------------
# Helpers: index / coords
# -------------------------
def idx_to_coord(idx):
    """Flatten index 0..NUM_NODES-1 -> (x,y)"""
    x = idx % GRID_W
    y = idx // GRID_W
    return x, y

def coord_to_idx(x, y):
    return int(y * GRID_W + x)

def in_bounds(x, y):
    return 0 <= x < GRID_W and 0 <= y < GRID_H

def state_index(x_idx, y_idx, cong, dir_idx):
    """Pack (x (0..15), y (0..15), cong (0..3), dir (0..3)) -> 0..NUM_STATES-1"""
    return ((((int(x_idx) * NY) + int(y_idx)) * NC + int(cong)) * ND + int(dir_idx))

# -------------------------
# Links precompute
# -------------------------
ALL_LINKS = set()
for x in range(GRID_W):
    for y in range(GRID_H):
        a = (x, y)
        for dx, dy in DELTAS.values():
            b = (x + dx, y + dy)
            if in_bounds(*b):
                ALL_LINKS.add(frozenset({a, b}))
ALL_LINKS = list(ALL_LINKS)

# -------------------------
# Packet class
# -------------------------
import uuid
class Packet:
    def __init__(self, src_idx, dst_idx):
        self.id = str(uuid.uuid4())[:8]
        self.src_idx = int(src_idx)
        self.dst_idx = int(dst_idx)
        self.node = Node()
        # set Node coords to src
        sx, sy = idx_to_coord(self.src_idx)
        self.node.x, self.node.y = sx, sy
        self.age = 0
        self.cum_reward = 0.0
        self.done = False

    @property
    def pos(self):
        return (self.node.x, self.node.y)

    @property
    def idx(self):
        return coord_to_idx(self.node.x, self.node.y)

# -------------------------
# Congestion & dir computation
# -------------------------
def nodes_in_neigh(center, radius=1):
    cx, cy = center
    nodes = []
    for x in range(GRID_W):
        for y in range(GRID_H):
            if abs(cx - x) + abs(cy - y) <= radius:
                nodes.append((x, y))
    return nodes

def incident_links(node):
    x, y = node
    links = []
    for dx, dy in DELTAS.values():
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            links.append(frozenset({(x, y), (nx, ny)}))
    return links

def compute_cong_dir_at(node_pos, node_visit_count, link_visit_count, faulty_nodes, faulty_links, active_packets=None):
    """
    Returns cong in {0,1,2,3} and dir in {0..3} for node at node_pos=(x,y).
    cong=3 means fault (either node faulty or a faulty outgoing link detected).
    dir is the dominant problematic outgoing direction:
      - if any outgoing link/neighbor faulty -> first faulty port (priority UP,RIGHT,DOWN,LEFT)
      - else direction with highest occupancy (neighbour visit proxy)
    """
    x, y = node_pos
    # per-port occupancy proxies and fault flags
    occ = {}
    fault = {}
    for i in range(NUM_ACTIONS):
        dx, dy = DELTAS[i]
        nx, ny = x + dx, y + dy
        if not in_bounds(nx, ny):
            occ[i] = 1.0   # treat off-grid as unavailable / high occupancy
            fault[i] = True
            continue
        neigh = (nx, ny)
        # occupancy estimate: normalized node_visit_count in neighbor
        occ_val = min(1.0, node_visit_count.get(neigh, 0) / HISTORY_SCALE)
        occ[i] = occ_val
        link = frozenset({(x, y), neigh})
        fault[i] = (neigh in faulty_nodes) or (link in faulty_links)

    # if current node itself is faulty -> cong=3 and dir pick any valid port
    if (x, y) in faulty_nodes:
        cong = 3
        # choose first in-bounds port as dir
        for i in range(NUM_ACTIONS):
            nx, ny = x + DELTAS[i][0], y + DELTAS[i][1]
            if in_bounds(nx, ny):
                return 3, i
        return 3, 0

    # if any outgoing port faulty -> cong=3 and dir = first faulty port (priority order)
    for i in range(NUM_ACTIONS):
        if fault[i]:
            return 3, i

    # otherwise compute avg occupancy and cong levels
    avg_occ = sum(occ.values()) / NUM_ACTIONS
    if avg_occ < 0.33:
        cong = 0
    elif avg_occ < 0.66:
        cong = 1
    else:
        cong = 2

    # dir: argmax occupancy (tie break by low index)
    dir_idx = max(range(NUM_ACTIONS), key=lambda a: (occ[a], -a))
    return cong, dir_idx

# -------------------------
# Neighbourhood congestion score
# -------------------------
def neighbourhood_congestion_score(center_pos, node_visit_count, link_visit_count, active_packets, radius=1):
    """
    Returns a combined neighbourhood congestion score in [0,1] that mixes history and immediate active occupancy.
    """
    neigh_nodes = nodes_in_neigh(center_pos, radius)
    node_area = max(1, len(neigh_nodes))
    total_hist = sum(node_visit_count.get(n, 0) for n in neigh_nodes)
    avg_hist = total_hist / node_area
    hist_score = float(np.tanh(avg_hist / HISTORY_SCALE))

    active_in_neigh = 0
    if active_packets is not None:
        for p in active_packets:
            px, py = p.pos
            if abs(px - center_pos[0]) + abs(py - center_pos[1]) <= radius:
                active_in_neigh += 1
    active_scale = max(1.0, node_area * 0.5)
    active_score = float(np.tanh(active_in_neigh / active_scale))

    W_HIST, W_ACTIVE = 0.6, 0.4
    combined = W_HIST * hist_score + W_ACTIVE * active_score
    return float(np.clip(combined, 0.0, 1.0)), {'hist': hist_score, 'active': active_score, 'active_in_neigh': active_in_neigh}

# -------------------------
# Reward function (multi-part)
# -------------------------
MAX_GRID_DIST = (GRID_W - 1) + (GRID_H - 1)  # max manhattan distance in 4x4 is 6

def reward_components(prev_pos, new_pos, dst_pos, was_blocked, cong_level, node_visit_count, link_visit_count, active_packets):
    """
    Compute reward as sum of components:
      R = R_move + R_dir + R_fault + R_local_cong + R_neigh + R_dest
    Returns (total_reward, details_dict)
    """
    details = {}

    # 1) base move penalty
    stayed = (prev_pos == new_pos)
    base_move_cost = MOVE_PENALTY + 10 if stayed else MOVE_PENALTY
    details['R_move'] = -W_MOVE * base_move_cost

    # 2) directional progress (normalized)
    prev_d = abs(prev_pos[0] - dst_pos[0]) + abs(prev_pos[1] - dst_pos[1])
    new_d = abs(new_pos[0] - dst_pos[0]) + abs(new_pos[1] - dst_pos[1])
    progress = prev_d - new_d
    norm = MAX_GRID_DIST if MAX_GRID_DIST > 0 else 1
    details['R_dir'] = W_DIR * (progress / norm)

    # 3) fault penalty
    if was_blocked or cong_level == 3:
        details['R_fault'] = -W_FAULT
    else:
        details['R_fault'] = 0.0

    # 4) local congestion penalty (node + link counts capped & normalized)
    node_count = min(node_visit_count.get(new_pos, 0), MAX_NODE_VISIT_CAP)
    # find link traversed
    traversed_link = frozenset({prev_pos, new_pos}) if prev_pos != new_pos else None
    link_count = min(link_visit_count.get(traversed_link, 0) if traversed_link is not None else 0, MAX_LINK_VISIT_CAP)
    # normalized
    node_norm = node_count / MAX_NODE_VISIT_CAP
    link_norm = link_count / MAX_LINK_VISIT_CAP
    details['R_local_cong'] = -W_LOCAL_CONG * (node_norm + link_norm)

    # 5) neighbourhood congestion penalty
    neigh_score, neigh_info = neighbourhood_congestion_score(new_pos, node_visit_count, link_visit_count, active_packets, radius=1)
    details.update({'neigh_hist': neigh_info['hist'], 'neigh_active': neigh_info['active'], 'neigh_active_count': neigh_info['active_in_neigh']})
    details['R_neigh'] = -W_NEIGH_CONG * neigh_score

    # 6) destination reward
    details['R_dest'] = DEST_REWARD if new_pos == dst_pos else 0.0

    total = sum(details[k] for k in ['R_move','R_dir','R_fault','R_local_cong','R_neigh','R_dest'])
    # clip to R_MAX
    total = float(np.clip(total, -R_MAX, R_MAX))
    details['total'] = total
    details['prev_dist'] = prev_d
    details['new_dist'] = new_d
    details['progress'] = progress
    details['cong_level'] = cong_level
    return total, details

# -------------------------
# Fault generation
# -------------------------
def generate_random_faults(max_node_faults=2, max_link_faults=2, avoid_nodes=None):
    nodes = [(x, y) for x in range(GRID_W) for y in range(GRID_H)]
    if avoid_nodes:
        nodes = [n for n in nodes if n not in avoid_nodes]
    node_fault_count = random.randint(0, max_node_faults)
    faulty_nodes = set(random.sample(nodes, k=node_fault_count)) if node_fault_count > 0 else set()
    # link faults
    possible_links = [L for L in ALL_LINKS if not (avoid_nodes and any(node in avoid_nodes for node in L))]
    link_fault_count = random.randint(0, max_link_faults)
    faulty_links = set(random.sample(possible_links, k=link_fault_count)) if link_fault_count > 0 else set()
    return faulty_nodes, faulty_links

# -------------------------
# Training loop (multi-packet)
# -------------------------
episode_rewards = []
total_steps = []

# global/shared congestion counters (persist across episode but decayed after each ep)
node_visit_count = defaultdict(float)
link_visit_count = defaultdict(float)

for episode in range(HM_EPISODES):
    # prepare per-episode items
    # choose representative avoid set to avoid marking initial start/dest faulty; we'll use random injection later
    avoid_nodes = None

    faulty_nodes, faulty_links = generate_random_faults(max_node_faults=2, max_link_faults=2, avoid_nodes=avoid_nodes)

    active_packets = []
    episode_reward = 0.0
    timestep = 0

    # optionally seed environment with a couple of packets initially
    # inject one or two to start
    for _ in range(1):
        s = random.randrange(NUM_NODES)
        d = random.randrange(NUM_NODES)
        while d == s:
            d = random.randrange(NUM_NODES)
        active_packets.append(Packet(s, d))

    while timestep < MAX_TIMESTEPS:
        timestep += 1

        # attempt injection
        if len(active_packets) < MAX_ACTIVE_PACKETS and random.random() < PACKET_INJECTION_PROB:
            s = random.randrange(NUM_NODES)
            d = random.randrange(NUM_NODES)
            while d == s:
                d = random.randrange(NUM_NODES)
            active_packets.append(Packet(s, d))

        if len(active_packets) == 0 and timestep > 10 and episode > 10:
            break

        # per-timestep structures
        occupied_links = set()
        random.shuffle(active_packets)

        # step each packet once
        for pkt in list(active_packets):
            if pkt.done:
                try:
                    active_packets.remove(pkt)
                except ValueError:
                    pass
                continue

            prev = pkt.pos
            cur_idx = pkt.idx
            dst_idx = pkt.dst_idx
            dst_coord = idx_to_coord(dst_idx)

            # compute cong & dir at current node (state features)
            cong_level, dir_idx = compute_cong_dir_at(prev, node_visit_count, link_visit_count, faulty_nodes, faulty_links, active_packets)

            # build state index: x=current node index, y=destination node index
            s_idx = state_index(cur_idx, dst_idx, cong_level, dir_idx)

            # action selection: epsilon-greedy on q_table
            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = int(np.argmax(q_table[s_idx]))

            # apply action to Packet.node (assumes Node.action(action) uses same mapping 0..3)
            prev_pos = (pkt.node.x, pkt.node.y)
            pkt.node.action(action)
            new_pos = (pkt.node.x, pkt.node.y)
            traversed_link = frozenset({prev_pos, new_pos}) if prev_pos != new_pos else None

            # detect blocked move (due to fault or boundary or other conditions)
            was_blocked = False
            if new_pos == prev_pos:
                # check if intended neighbor was faulty or off-grid
                dx, dy = DELTAS[action]
                intended = (prev_pos[0] + dx, prev_pos[1] + dy)
                if (not in_bounds(*intended)) or (intended in faulty_nodes) or (frozenset({prev_pos, intended}) in faulty_links):
                    was_blocked = True
            # link contention: if another packet already occupied this link this timestep -> blocked
            if traversed_link is not None and traversed_link in occupied_links:
                # revert
                pkt.node.x, pkt.node.y = prev_pos
                new_pos = prev_pos
                traversed_link = None
                was_blocked = True

            # if blocked by faults or contention, optionally increase local counters for attempted congestion
            if was_blocked:
                # increase a small penalty to node_visit_count (reflect attempted congestion)
                node_visit_count[prev_pos] += 0.2
            else:
                # move succeeded: mark link occupied and increment counters
                if traversed_link is not None:
                    occupied_links.add(traversed_link)
                    link_visit_count[traversed_link] += 1.0
                node_visit_count[new_pos] += 1.0

            # compute reward
            r, details = reward_components(prev, new_pos, dst_coord, was_blocked, cong_level, node_visit_count, link_visit_count, active_packets)
            pkt.cum_reward += r
            episode_reward += r
            pkt.age += 1

            # get next state (post-action)
            new_cong, new_dir = compute_cong_dir_at(new_pos, node_visit_count, link_visit_count, faulty_nodes, faulty_links, active_packets)
            s2_idx = state_index(coord_to_idx(new_pos[0], new_pos[1]), dst_idx, new_cong, new_dir)

            # Q-learning update
            max_future = np.max(q_table[s2_idx])
            current_q = q_table[s_idx, action]
            if new_pos == dst_coord:
                target = DEST_REWARD
                pkt.done = True
            else:
                target = r + DISCOUNT * max_future
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * target
            q_table[s_idx, action] = new_q

            # remove delivered or aged packets
            if pkt.done or pkt.age > PACKET_MAX_AGE:
                if pkt in active_packets:
                    active_packets.remove(pkt)

        # epsilon decay and small housekeeping
        if episode > 10:
            EPSILON *= EPS_DECAY

        # debugging / optional visualization (omitted) ...

    # end of episode bookkeeping
    episode_rewards.append(episode_reward)
    total_steps.append(timestep)

    # decay congestion counters to avoid runaway
    if USE_DECAY:
        for k in list(node_visit_count.keys()):
            node_visit_count[k] *= CONGESTION_DECAY
            if node_visit_count[k] < 1e-3:
                del node_visit_count[k]
        for k in list(link_visit_count.keys()):
            link_visit_count[k] *= CONGESTION_DECAY
            if link_visit_count[k] < 1e-3:
                del link_visit_count[k]

    # occasional logging
    if episode % 200 == 0:
        recent = np.mean(episode_rewards[-200:]) if len(episode_rewards) >= 200 else np.mean(episode_rewards)
        print(f"Episode {episode} | eps {EPSILON:.4f} | recent avg reward {recent:.2f} | timesteps {timestep}")

# Save Q-table
with open(f"qtable_4x4_absstate_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training finished. Q-table saved.")
