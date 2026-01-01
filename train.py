# train.py
import numpy as np
import random
import time
import pickle
from collections import defaultdict

# from fault_training import faulty_nodes
from new_environment import NetworkEnv, Packet

# -------------------------
# Training & env hyperparams
# -------------------------
GRID_W = GRID_H = 5
NUM_VCS = 4

# instantiate environment
env = NetworkEnv(grid_w=GRID_W, grid_h=GRID_H, num_vcs=NUM_VCS,
                 vc_fault_persistence=1, vc_clear_persistence=1, vc_ema_alpha=0.25,
                 history_scale=50.0)

NUM_NODES = env.NUM_NODES
NUM_ACTIONS = env.NUM_ACTIONS

# state dims: (row, col, destination_idx, cong(4), dir(4))
NC = 4
ND = 4
NX = GRID_H  # rows
NY = GRID_W  # cols
NUM_FAULTY = 7

# NUM_STATES = (NX * NY) * NUM_NODES * NC * ND  # current node (row,col) x destination x cong x dir

NUM_STATES = 4 * 2 ** NUM_FAULTY

# Q-table
q_table = np.random.uniform(-1, 0, size=(NUM_STATES, NUM_ACTIONS)).astype(np.float32)
print(f"Q-table shape: {q_table.shape}, bytes: {q_table.nbytes} (~{q_table.nbytes/1024:.1f} KB)")

# RL hyperparams
HM_EPISODES = 40000
EXPLORATION_EPISOPES = 5000
MAX_TIMESTEPS = 400

LEARNING_RATE = 0.0002
DISCOUNT = 0.95

EPSILON = 0.95
EPS_DECAY = 0.99985

# multi-packet params
MAX_ACTIVE_PACKETS = 40
PACKET_INJECTION_PROB = 1
PACKET_MAX_AGE = 400

# reward hyperparams (keep in sync with env if changed)
MOVE_PENALTY = 4
DEST_REWARD = 200
W_MOVE = 3.0
W_DIR = 10.0
W_FAULT = 30.0
W_LOCAL_CONG = 4.0
W_NEIGH_CONG = 20.0
W_NEIGH_FAULT = 10.0
MAX_NODE_VISIT_CAP = 50.0
MAX_LINK_VISIT_CAP = 50.0
R_MAX = 1000.0

# -------------------------
# helpers for state index
# -------------------------
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
        [(0,2),(1,1),(2,0)],
        [(2,0),(1,-1),(0,-2)],
        [(0,-2),(-1,-1),(-2,0)],
        [(-2,0),(-1,1),(0,2)]
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
    packed = (dir << 7) | bits
    # optional: return components for debugging
    return packed  # or return dir, bits, packed

# -------------------------
# reward function (wraps env methods)
# -------------------------
def compute_reward(prev_pos, new_pos, dst_pos, blocked, cong_level, env_obj, active_packets):
    """
    Compose reward using env's counters and neighbourhood functions.
    """
    details = {}
    # base move
    stayed = (prev_pos == new_pos)
    base_move_cost = MOVE_PENALTY + 5 if stayed else MOVE_PENALTY
    details['R_move'] = -W_MOVE * base_move_cost

    # print("R_move", details['R_move'])

    # directional progress
    prev_d = abs(prev_pos[0] - dst_pos[0]) + abs(prev_pos[1] - dst_pos[1])
    new_d = abs(new_pos[0] - dst_pos[0]) + abs(new_pos[1] - dst_pos[1])
    progress = prev_d - new_d
    max_grid_dist = (GRID_W - 1) + (GRID_H - 1)
    if progress > 0:
        progress *= 3
    details['R_dir'] = W_DIR * (progress / max(1, max_grid_dist))

    # print("R_dir", details['R_dir'])

    details['R_busy_link'] = -1.0 if blocked['busy_link'] else 0.0
    details['R_no_vc'] = -1.0 if blocked['no_vc'] else 0.0
    details['R_oob'] = -1.0 if blocked['oob'] else 0.0
    details["R_cong"] = -15.0 if (blocked['sustained'] or cong_level == 3) else 0.0

    # fault penalty
    details['R_fault'] = -W_FAULT if (blocked['faulty']) else 0.0

    # print("R_fault", details['R_fault'])

    # local congestion penalty
    # cong_score, fault_score, cong_info = env_obj.neighbourhood_congestion(prev_pos, active_packets=active_packets, radius=1)
    # details.update({'node_hist': cong_info['hist'], 'node_active': cong_info['active'], 'node_fault': cong_info['fault'],
    #                 'node_active_count': cong_info['active_in_neigh']})
    # details['R_local_cong'] = -W_LOCAL_CONG * cong_score
    # details['R_neigh_fault'] = -W_NEIGH_FAULT * fault_score

    # print("R_local_cong", details['R_local_cong'])

    # neighbourhood congestion
    neigh_score, neigh_fault_score, neigh_info = env_obj.neighbourhood_congestion(new_pos, active_packets=active_packets, radius=1)
    details.update({'neigh_hist': neigh_info['hist'], 'neigh_active': neigh_info['active'], 'neigh_fault': neigh_info['fault'], 'neigh_active_count': neigh_info['active_in_neigh']})
    details['R_neigh_cong'] = -W_NEIGH_CONG * neigh_score
    details['R_neigh_fault'] = -W_NEIGH_FAULT * neigh_fault_score

    # print("R_neigh", details['R_neigh'])
    # print("R_neigh_fault", details['R_neigh_fault'])

    # dest reward
    details['R_dest'] = DEST_REWARD if new_pos == dst_pos else 0.0

    # print("R_dest", details['R_dest'])

    total = sum(details[k] for k in ['R_move','R_dir','R_fault','R_neigh_cong','R_dest','R_neigh_fault', 'R_busy_link', 'R_oob','R_cong','R_no_vc'])
    total = float(np.clip(total, -R_MAX, R_MAX))
    details['total'] = total
    details['prev_dist'] = prev_d
    details['new_dist'] = new_d
    details['progress'] = progress
    details['cong_level'] = cong_level

    # print("R_total", details['total'])
    return total, details

# -------------------------
# Training loop
# -------------------------
episode_rewards = []
total_steps = []
avg_packet_age = []
avg_injected = []
avg_delivered = []
avg_dropped = []
avg_undelivered = []

for episode in range(HM_EPISODES):
    # print("Episode", episode)
    # reset per-episode env counters if desired (vc_free etc.)
    env.reset_counters()


    # pick initial seed packet(s)
    active_packets = []

    episode_reward = 0.0
    timestep = 0
    injected = 0
    delivered = 0
    dropped = 0
    packet_age = 0

    s = random.randrange(NUM_NODES)
    d = random.randrange(NUM_NODES)
    while d == s:
        d = random.randrange(NUM_NODES)
    # ensure avoid when generating faults
    s_coord = env.idx_to_coord(s)
    d_coord = env.idx_to_coord(d)
    avoid_nodes = {s_coord, d_coord}

    # generate faults (env will store them)
    env.generate_random_faults(max_node_faults=4, max_link_faults=0, avoid_nodes=avoid_nodes)

    # inject initial packet if VC available
    p0 = env.try_inject(s, d)
    if p0:
        injected += 1
        active_packets.append(p0)

    while timestep < MAX_TIMESTEPS:
        # print("sustained faulty", env.sustained_faulty_ports)
        timestep += 1
        env.start_timestep()

        # injection attempts
        if len(active_packets) < MAX_ACTIVE_PACKETS and random.random() < PACKET_INJECTION_PROB:

            s = random.randrange(NUM_NODES)
            d = random.randrange(NUM_NODES)

            while env.idx_to_coord(d) in env.faulty_nodes:
                d = random.randrange(NUM_NODES)
            while env.idx_to_coord(s) in env.faulty_nodes:
                s = random.randrange(NUM_NODES)
            while d == s:
                d = random.randrange(NUM_NODES)

            # skip inject if node explicitly faulty or sustained faulty
            sr, sc = env.idx_to_coord(s)
            if (sr, sc) in env.sustained_faulty_ports:
                pass
            else:
                pnew = env.try_inject(s, d)
                if pnew:
                    injected += 1
                    active_packets.append(pnew)

        # if len(active_packets) == 0 and timestep > 10 and episode > 10:
        #     break

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
            neighbor_cong = env.compute_cong_dir_at(prev, active_packets=active_packets)
            faulty_nodes = env.faulty_at_start(prev, neighbor_cong)

            # build state index
            s_idx = state_index(env.coord_to_idx(prev[0], prev[1]), pkt.dst_idx, faulty_nodes, N=GRID_W)

            # select action (epsilon-greedy)
            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = int(np.argmax(q_table[s_idx]))

            # attempt move and commit
            blocked, traversed_link, intended = env.attempt_and_commit_move(pkt, action)

            new_pos = pkt.pos

            # if packet reached destination, mark done and release VC at dest (packet leaves)
            if new_pos == dst_coord:
                pkt.done = True
                # release VC at destination (packet delivered)
                # env.release_vc_at(new_pos)
                # pkt.vc holds (node,port) which the packet currently occupies (or None)
                env.release_vc_at(pkt.vc if pkt.vc is not None else new_pos)
                pkt.vc = None

                if pkt in active_packets:
                    active_packets.remove(pkt)
                delivered += 1
                packet_age += pkt.age

            # compute reward
            r, details = compute_reward(prev, new_pos, dst_coord, blocked, neighbor_cong[action], env, active_packets)

            pkt.cum_reward += r
            episode_reward += r
            pkt.age += 1

            # next state
            neighbor_cong = env.compute_cong_dir_at(new_pos, active_packets=active_packets)
            faulty_nodes = env.faulty_at_start(new_pos, neighbor_cong)
            s2_idx = state_index(env.coord_to_idx(new_pos[0], new_pos[1]), pkt.dst_idx, faulty_nodes, N=GRID_W)

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
                dropped += 1
                packet_age += pkt.age

        # end of timestep housekeeping
        env.end_timestep()
    for packet in active_packets:
        packet_age += packet.age
    # epsilon decay
    if episode > 10:
        EPSILON *= EPS_DECAY

    # end episode
    episode_rewards.append(episode_reward)
    total_steps.append(timestep)
    avg_packet_age.append(packet_age / max(1, injected))
    avg_injected.append(injected)
    avg_dropped.append(dropped)
    avg_delivered.append(delivered)
    avg_undelivered.append(len(active_packets))

    # print("Delivered", delivered)
    # print("Dropped", dropped)
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
        recent_avg_age = np.mean(avg_packet_age[-200:]) if len(avg_packet_age) >= 200 else np.mean(avg_packet_age)
        recent_avg_inj = np.mean(avg_injected[-200:]) if len(avg_injected) >= 200 else np.mean(avg_injected)
        recent_avg_del = np.mean(avg_delivered[-200:]) if len(avg_delivered) >= 200 else np.mean(avg_delivered)
        recent_avg_drp = np.mean(avg_dropped[-200:]) if len(avg_dropped) >= 200 else np.mean(avg_dropped)
        recent_avg_undel = np.mean(avg_undelivered[-200:]) if len(avg_undelivered) >= 200 else np.mean(avg_undelivered)
        print(f"Episode {episode} | eps {EPSILON:.4f} | recent avg reward {recent:.2f} | timesteps {timestep}")
        print(
            f"Episode: {episode}, Injected: {recent_avg_inj:.1f}, Delivered: {recent_avg_del:.1f}, "
            f"Dropped: {recent_avg_drp:.1f}, Active: {recent_avg_undel:.1f}, Avg Packet Age: {recent_avg_age:.1f}")

# Save Q-table
    if episode >= 20000 and episode % 10000 == 0:
        with open(f"qtable_grid{GRID_W}x{GRID_H}_vcs{NUM_VCS}_{int(time.time())}.pkl", "wb") as f:
            pickle.dump(q_table, f)

        print("Training finished. Q-table saved.")
