# train.py
import numpy as np
import random
import time
import pickle
from collections import defaultdict
from updated_env import NetworkEnv, Packet

# -------------------------
# Training & env hyperparams
# -------------------------
GRID_W = GRID_H = 5
NUM_VCS = 2

# instantiate environment
env = NetworkEnv(grid_w=GRID_W, grid_h=GRID_H, num_vcs=NUM_VCS,
                 vc_fault_persistence=5, vc_clear_persistence=3, vc_ema_alpha=0.25,
                 history_scale=8.0)

NUM_NODES = env.NUM_NODES
NUM_ACTIONS = env.NUM_ACTIONS

# state dims: (row, col, destination_idx, cong(4), dir(4))
NC = 4
ND = 4
NX = GRID_H  # rows
NY = GRID_W  # cols

NUM_STATES = (NX * NY) * NUM_NODES * NC * ND  # current node (row,col) x destination x cong x dir

# Q-table
q_table = np.random.uniform(-1, 0, size=(NUM_STATES, NUM_ACTIONS)).astype(np.float32)
print(f"Q-table shape: {q_table.shape}, bytes: {q_table.nbytes} (~{q_table.nbytes/1024:.1f} KB)")

# RL hyperparams
HM_EPISODES = 3000
MAX_TIMESTEPS = 400

LEARNING_RATE = 0.1
DISCOUNT = 0.95

EPSILON = 0.95
EPS_DECAY = 0.9995

# multi-packet params
MAX_ACTIVE_PACKETS = 6
PACKET_INJECTION_PROB = 0.25
PACKET_MAX_AGE = 200

# reward hyperparams (keep in sync with env if changed)
MOVE_PENALTY = 10
DEST_REWARD = 50
W_MOVE = 1.0
W_DIR = 20.0
W_FAULT = 60.0
W_LOCAL_CONG = 5.0
W_NEIGH_CONG = 30.0
MAX_NODE_VISIT_CAP = 50.0
MAX_LINK_VISIT_CAP = 50.0
R_MAX = 1000.0

# -------------------------
# helpers for state index
# -------------------------
def state_index_from(row, col, dst_idx, cong, dir_idx):
    """Pack (row,col,dst,cong,dir) into linear index."""
    node_index = row * GRID_W + col
    # order: node_index (0..NUM_NODES-1), destination (0..NUM_NODES-1), cong, dir
    base = (node_index * NUM_NODES) + dst_idx
    return ((((base) * NC) + int(cong)) * ND) + int(dir_idx)

def state_index(source, destination, faulty_routers, N=GRID_W):
    # source/destination: (row, col)
    sx, sy = env.idx_to_coord(source)
    dx, dy = env.idx_to_coord(destination)
    delx = dx - sx
    dely = dy - sy

    # dir mapping: 0=right,1=down,2=left,3=up
    if dely > 0:
        dir = 0 if delx >= 0 else 3
    elif dely == 0:
        dir = 1 if delx > 0 else 3
    else:  # dely < 0
        dir = 1 if delx > 0 else 2
    # print("dir", dir)
    # neighbors: right, down, left, up
    deltas = [(0,1),(1,0),(0,-1),(-1,0)]
    delta_th = [
        [(0,2),(1,1),(2,0),(1,2)],
        [(2,0),(1,-1),(0,-2),(2,-1)],
        [(0,-2),(-1,-1),(-2,0),(-1,-2)],
        [(-2,0),(-1,1),(0,2),(-2,1)]
    ]

    faulty_set = set(faulty_routers)
    def is_fault(p):
        if N is not None:
            r,c = p
            if not (0 <= r < N and 0 <= c < N):
                return 1
        return 1 if p in faulty_set else 0
    neigh = []
    bits = 0
    for off in deltas:
        bits = (bits << 1) | is_fault((sx + off[0], sy + off[1]))
        neigh.append(is_fault((sx + off[0], sy + off[1])))
    for off in delta_th[dir]:
        bits = (bits << 1) | is_fault((sx + off[0], sy + off[1]))
        neigh.append(is_fault((sx + off[0], sy + off[1])))
    # print("NEigh", neigh)
    packed = (dir << 8) | bits
    # optional: return components for debugging
    return packed  # or return dir, bits, packed

# -------------------------
# reward function (wraps env methods)
# -------------------------
def compute_reward(prev_pos, new_pos, dst_pos, was_blocked, cong_level, env_obj, active_packets):
    """
    Compose reward using env's counters and neighbourhood functions.
    """
    details = {}
    # base move
    stayed = (prev_pos == new_pos)
    base_move_cost = MOVE_PENALTY + 10 if stayed else MOVE_PENALTY
    details['R_move'] = -W_MOVE * base_move_cost

    # directional progress
    prev_d = abs(prev_pos[0] - dst_pos[0]) + abs(prev_pos[1] - dst_pos[1])
    new_d = abs(new_pos[0] - dst_pos[0]) + abs(new_pos[1] - dst_pos[1])
    progress = prev_d - new_d
    max_grid_dist = (GRID_W - 1) + (GRID_H - 1)
    details['R_dir'] = W_DIR * (progress / max(1, max_grid_dist))

    # fault penalty
    details['R_fault'] = -W_FAULT if (was_blocked or cong_level == 3) else 0.0

    # local congestion penalty
    node_count = min(env_obj.node_visit_count.get(new_pos, 0.0), MAX_NODE_VISIT_CAP)
    traversed_link = frozenset({prev_pos, new_pos}) if prev_pos != new_pos else None
    link_count = min(env_obj.link_visit_count.get(traversed_link, 0.0) if traversed_link is not None else 0.0, MAX_LINK_VISIT_CAP)
    node_norm = node_count / MAX_NODE_VISIT_CAP
    link_norm = link_count / MAX_LINK_VISIT_CAP
    details['R_local_cong'] = -W_LOCAL_CONG * (node_norm + link_norm)

    # neighbourhood congestion
    neigh_score, neigh_info = env_obj.neighbourhood_congestion(new_pos, active_packets=active_packets, radius=1)
    details.update({'neigh_hist': neigh_info['hist'], 'neigh_active': neigh_info['active'], 'neigh_active_count': neigh_info['active_in_neigh']})
    details['R_neigh'] = -W_NEIGH_CONG * neigh_score

    # dest reward
    details['R_dest'] = DEST_REWARD if new_pos == dst_pos else 0.0

    total = sum(details[k] for k in ['R_move','R_dir','R_fault','R_local_cong','R_neigh','R_dest'])
    total = float(np.clip(total, -R_MAX, R_MAX))
    details['total'] = total
    details['prev_dist'] = prev_d
    details['new_dist'] = new_d
    details['progress'] = progress
    details['cong_level'] = cong_level
    return total, details

# -------------------------
# Training loop
# -------------------------
episode_rewards = []
total_steps = []

for episode in range(HM_EPISODES):
    # reset per-episode env counters if desired (vc_free etc.)
    env.reset_counters()

    # pick initial seed packet(s)
    active_packets = []
    s = random.randrange(NUM_NODES)
    d = random.randrange(NUM_NODES)
    while d == s:
        d = random.randrange(NUM_NODES)
    # ensure avoid when generating faults
    s_coord = env.idx_to_coord(s)
    d_coord = env.idx_to_coord(d)
    avoid_nodes = {s_coord, d_coord}

    # generate faults (env will store them)
    env.generate_random_faults(max_node_faults=2, max_link_faults=0, avoid_nodes=avoid_nodes)

    # inject initial packet if VC available
    p0 = env.try_inject(s, d)
    if p0:
        active_packets.append(p0)

    episode_reward = 0.0
    timestep = 0

    while timestep < MAX_TIMESTEPS:
        timestep += 1
        env.start_timestep()

        # injection attempts
        if len(active_packets) < MAX_ACTIVE_PACKETS and random.random() < PACKET_INJECTION_PROB:
            s = random.randrange(NUM_NODES)
            d = random.randrange(NUM_NODES)
            while d == s:
                d = random.randrange(NUM_NODES)
            # skip inject if node explicitly faulty or sustained faulty
            sr, sc = env.idx_to_coord(s)
            if (sr, sc) in env.faulty_nodes or (sr, sc) in env.sustained_faulty:
                pass
            else:
                pnew = env.try_inject(s, d)
                if pnew:
                    active_packets.append(pnew)

        if len(active_packets) == 0 and timestep > 10 and episode > 10:
            break

        # per-timestep service order
        random.shuffle(active_packets)

        for pkt in list(active_packets):
            if pkt.done:
                if pkt in active_packets:
                    active_packets.remove(pkt)
                continue

            prev = pkt.pos
            dst_coord = env.idx_to_coord(pkt.dst_idx)

            # compute cong & dir at current node (features)
            cong_level, dir_idx = env.compute_cong_dir_at(prev, active_packets=active_packets)

            # build state index
            s_idx = state_index(env.coord_to_idx(prev[0], prev[1]), pkt.dst_idx, env.faulty_nodes, N=GRID_W)

            # select action (epsilon-greedy)
            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = int(np.argmax(q_table[s_idx]))

            # attempt move and commit
            was_blocked, traversed_link, intended = env.attempt_and_commit_move(pkt, action)

            new_pos = pkt.pos

            # if packet reached destination, mark done and release VC at dest (packet leaves)
            if new_pos == dst_coord:
                pkt.done = True
                # release VC at destination (packet delivered)
                env.release_vc_at(new_pos)
                if pkt in active_packets:
                    active_packets.remove(pkt)

            # compute reward
            r, details = compute_reward(prev, new_pos, dst_coord, was_blocked, cong_level, env, active_packets)
            pkt.cum_reward += r
            episode_reward += r
            pkt.age += 1

            # next state
            new_cong, new_dir = env.compute_cong_dir_at(new_pos, active_packets=active_packets)
            s2_idx = state_index_from(new_pos[0], new_pos[1], pkt.dst_idx, new_cong, new_dir)

            # Q-learning update
            max_future = np.max(q_table[s2_idx])
            current_q = q_table[s_idx, action]
            if pkt.done:
                target = DEST_REWARD
            else:
                target = r + DISCOUNT * max_future
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * target
            q_table[s_idx, action] = new_q

            # remove aged packets
            if pkt.age > PACKET_MAX_AGE:
                # release VC at current position
                env.release_vc_at(pkt.pos)
                if pkt in active_packets:
                    active_packets.remove(pkt)

        # end of timestep housekeeping
        env.end_timestep()

        # epsilon decay
        if episode > 10:
            EPSILON *= EPS_DECAY

    # end episode
    episode_rewards.append(episode_reward)
    total_steps.append(timestep)

    # decay historical visit counters a bit (optional)
    for k in list(env.node_visit_count.keys()):
        env.node_visit_count[k] *= 0.95
        if env.node_visit_count[k] < 1e-3:
            del env.node_visit_count[k]
    for k in list(env.link_visit_count.keys()):
        env.link_visit_count[k] *= 0.95
        if env.link_visit_count[k] < 1e-3:
            del env.link_visit_count[k]

    if episode % 200 == 0:
        recent = np.mean(episode_rewards[-200:]) if len(episode_rewards) >= 200 else np.mean(episode_rewards)
        print(f"Episode {episode} | eps {EPSILON:.4f} | recent avg reward {recent:.2f} | timesteps {timestep}")

# Save Q-table
with open(f"qtable_grid{GRID_W}x{GRID_H}_vcs{NUM_VCS}_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training finished. Q-table saved.")
