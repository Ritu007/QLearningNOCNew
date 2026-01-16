# train.py
import numpy as np
import random
import time
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon
# from fault_training import faulty_nodes
from environment import NetworkEnv, Packet

# -------------------------
# Training & env hyperparams
# -------------------------
GRID_W = GRID_H = 16
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
MAX_FAULTS = 50
render = False


# NUM_STATES = (NX * NY) * NUM_NODES * NC * ND  # current node (row,col) x destination x cong x dir

NUM_STATES = NUM_ACTIONS * 4 * 4 * 4**2 * 2**2

qtable = "qtable_grid5x5_vcs4_1768334084.pkl"
# load q-table
if qtable is not None:
    with open(qtable, 'rb') as f:
        q_table = pickle.load(f)

else:
    # Q-table
    q_table = np.random.uniform(-1, 0, size=(NUM_STATES, NUM_ACTIONS)).astype(np.float32)
    print(f"Q-table shape: {q_table.shape}, bytes: {q_table.nbytes} (~{q_table.nbytes/1024:.1f} KB)")

# RL hyperparams
HM_EPISODES = 100001
# EXPLORATION_EPISOPES = 5000
MAX_TIMESTEPS = 1000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

EPSILON = 0.8
EPS_DECAY = 0.99997
EPS_MIN = 0.1

INJECTION_RATE = 0.02

# multi-packet params
# MAX_ACTIVE_PACKETS = int(INJECTION_RATE * (NUM_NODES ** 2))
MAX_ACTIVE_PACKETS = 1
PACKET_INJECTION_PROB = 1
PACKET_MAX_AGE = 300

print(MAX_ACTIVE_PACKETS)
# reward hyperparams (keep in sync with env if changed)
MOVE_PENALTY = 1
REVISIT_PENALTY = 1
BUSY_LINK_PENALTY = 1
DEST_REWARD = 1
FAULT_PENALTY = 1
DROP_PENALTY = 1


W_MOVE = 0.5
W_DIR = 0.2
W_FAULT = 1
W_CONG = 0.5
W_LOCAL_CONG = 0.0
W_NEIGH_CONG = 0.2
W_NEIGH_FAULT = 0.0
W_REVISIT = 0.8
W_AGE = 0.5
W_DEL = 5
MAX_NODE_VISIT_CAP = 50.0
MAX_LINK_VISIT_CAP = 50.0
R_MAX = 20.0


import math
from utils import draw_grid


def age_penalty(age, t_min=7, t_rise=10, t_mid=15.0, k=0.59):
    """
    Smooth age penalty in [0,1]:
      - minimal for age <= t_min (kept tiny)
      - starts rising around t_rise
      - midpoint (50%) at t_mid
      - steepness controlled by k (higher k => sharper rise)
    Returns a float in [0,1].
    """
    # logistic (sigmoid) core
    s = 1.0 / (1.0 + math.exp(-k * (age - t_mid)))
    if age <= t_min:
        # scale down to keep minimal penalty for small ages (continuous)
        scale = (age / float(t_min)) * 0.25  # tweak 0.25 to set how "minimal" it stays
        return s * scale
    return s

def count_two_hop_faults_quantized(sx, sy, dx, dy, faulty_routers):
    two_hops_neighbours = [(-1, 1), (0, 2), (1, 1), (2, 0),
                           (1, -1), (0, -2), (-1, -1), (-2, 0)]

    # raw scores
    counts = {'right': 0.0, 'left': 0.0, 'down': 0.0, 'up': 0.0}

    delx = dx - sx
    dely = dy - sy

    # print("del", delx, dely)

    # pick quadrant
    if delx >= 0 and dely > 0:
        dir_idx = 0
        axes = ('right', 'down')
    elif delx > 0 and dely <= 0:
        dir_idx = 1
        axes = ('down', 'left')
    elif delx <= 0 and dely < 0:
        dir_idx = 2
        axes = ('left', 'up')
    elif delx < 0 and dely >= 0:
        dir_idx = 3
        axes = ('up', 'right')
    else:
        # dir_idx = 0
        # print("Here", delx, dely)
        return 0, 0, 0, 0, 0

    offset = (dir_idx * 2) % 8
    ports = [[1,2,3],[2,3,0],[3,0,1],[0,1,2]]
    min_x, max_x = min(sx, dx), max(sx, dx)
    min_y, max_y = min(sy, dy), max(sy, dy)
    neigh_index = 0
    port = []
    port.extend(ports[dir_idx % 4])
    port.extend(ports[(dir_idx + 1) % 4])
    # print("Ports", dir_idx, port)
    for port_index in range(6):
        weight = 0.0
        ox, oy = two_hops_neighbours[(neigh_index + offset) % 8]
        nx, ny = sx + ox, sy + oy



        if (nx, ny) in faulty_routers or not (0 <= nx < GRID_H and 0 <= ny < GRID_W) or ((nx, ny), port[port_index]) in env.sustained_faulty_ports:
            critical = (min_x <= nx <= max_x) and (min_y <= ny <= max_y)
            weight = 1.0 if critical else 0.25

        # if ((nx, ny), port[port_index]) in env.sustained_faulty_ports:
        #     critical = (min_x <= nx <= max_x) and (min_y <= ny <= max_y)
        #     weight = 1.0 if critical else 0.25

            if port_index < 3:
                counts[axes[0]] += weight
            if neigh_index >= 3:
                counts[axes[1]] += weight

        # print("Neigh", (nx, ny), port[port_index], weight)
        # print("Axis 0:", counts[axes[0]], "Axis 1:", counts[axes[1]])

        if not (port_index == 2 and neigh_index == 2):
            neigh_index += 1


    # --- quantization step ---
    def quantize(v):
        if v >= 1.5:
            return 2
        elif v >= 0.5:
            return 1
        else:
            return 0

    return (dir_idx,
        quantize(counts['right']),
        quantize(counts['down']),
        quantize(counts['left']),
        quantize(counts['up']),
    )

# -------------------------
# helpers for state index
# -------------------------
def state_index(source, destination, prev_node, faulty_routers, N=GRID_W):
    # source/destination: (row, col)
    sx, sy = env.idx_to_coord(source)
    dx, dy = env.idx_to_coord(destination)
    delx = dx - sx
    dely = dy - sy

    prev_action = 0
    for key, value in env.DELTAS.items():
        if prev_node is None:
            break
        if (sx, sy) == (prev_node[0] + value[0], prev_node[1] + value[1]):
            prev_action = key

        # neighbors: right, down, left, up
    deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    delta_th = [
        [(0, 2), (1, 1), (2, 0)],
        [(2, 0), (1, -1), (0, -2)],
        [(0, -2), (-1, -1), (-2, 0)],
        [(-2, 0), (-1, 1), (0, 2)]
    ]

    # dir mapping: 0=right,1=down,2=left,3=up
    # if dely > 0:
    #     dir = 0 if delx >= 0 else 3
    # elif dely == 0:
    #     dir = 1 if delx > 0 else 3
    # else:  # dely < 0
    #     dir = 1 if delx > 0 else 2
    # print("dir", dir)
    # offset = (delx, dely)
    # rel_off = (offset[0] + N - 1, offset[1] + N - 1)
    # rel_dist = rel_off[0] * (2 * N - 1) + rel_off[1]

    # print(f"Offset: {offset}, Rel_off: {rel_off}, Rel Dist: {rel_dist}")
    hops = np.abs(delx) + np.abs(dely)

    direction, right, down, left, up = count_two_hop_faults_quantized(sx,sy,dx,dy,faulty_routers)

    # print(f"Dir: {direction}, R: {right}, D: {down}, L: {left}, U: {up}")

    two_hop_state = (0,0)
    one_hop_state = [0,0]

    two_hop_faults = [right, down, left, up]
    neighs = []
    for i in range(4):
        neighs.append((sx + deltas[i][0], sy + deltas[i][1]))

    one_hop_faults = [0, 0, 0, 0]
    for i in range(4):
        opp_port = (i + 2) % 4
        if neighs[i] in faulty_routers or (neighs[i], opp_port) in env.sustained_faulty_ports:
            one_hop_faults[i] = 1
            two_hop_faults[i] = 3
        if not (0 <= neighs[i][0] < GRID_H and 0 <= neighs[i][1] < GRID_W):
            one_hop_faults[i] = 1
            two_hop_faults[i] = 3

    two_hop_state = (two_hop_faults[direction % 4], two_hop_faults[(direction + 1) % 4])
    for i in range(direction + 2, direction + 4):
        if one_hop_faults[i % 4] == 1:
            one_hop_state[i % 2] = 1

    if hops <= 1:
        dist = 0
    elif hops == 2:
        dist = 1
    elif 3 <= hops <= 4:
        dist = 2
    else:
        dist = 3

    faulty_set = set(faulty_routers)
    def is_fault(p):
        if N is not None:
            r,c = p
            if not (0 <= r < N and 0 <= c < N):
                return 1
        return 1 if p in faulty_set else 0
    neigh = []
    bits = 0
    # for off in deltas:
    #     bits = (bits << 1) | is_fault((sx + off[0], sy + off[1]))
    #     neigh.append(is_fault((sx + off[0], sy + off[1])))
    # for off in delta_th[dir]:
    #     bits = (bits << 1) | is_fault((sx + off[0], sy + off[1]))
    #     neigh.append(is_fault((sx + off[0], sy + off[1])))
    # print("NEigh", neigh)

    for fault in one_hop_state:
        bits = (bits << 1) | fault
        # neigh.append(is_fault((sx + off[0], sy + off[1])))

    state = [prev_action, direction, dist, two_hop_state, one_hop_state]
    state.extend(neigh)
    combined = (((prev_action * 4 + direction) * 4 + dist) * 4 + two_hop_state[0]) * 4 + two_hop_state[1]
    packed = (combined << 2) | bits

    # combined = direction
    # state = [rel_dist]
    # print(f'Faults: {faulty_routers}, One hop: {one_hop_faults}, State_one: {one_hop_state}, Two Hop: {two_hop_faults}, State Two: {two_hop_state}')
    # print(f"Source: {(sx, sy)}, Destination: {(dx, dy)}, Prev: {prev_node}, Prev Action: {prev_action}, State: {state}, Index: {packed}")
    # optional: return components for debugging
    return packed  # or return dir, bits, packed

# -------------------------
# reward function (wraps env methods)
# -------------------------
def compute_reward(prev_pos, new_pos, dst_pos, blocked, cong_level, env_obj, active_packets, history, age):
    """
    Compose reward using env's counters and neighbourhood functions.
    """
    details = {}
    # base move

    base_move_cost = MOVE_PENALTY
    stayed = (prev_pos == new_pos)
    # if stayed:
    #     base_move_cost += REVISIT_PENALTY
    if blocked['stayed']:
        base_move_cost *= 2

    details['R_move'] = -W_MOVE * base_move_cost
    # revisit_penalty = 0.0
    # print(f"Base Move Cost {base_move_cost}, R move: {details['R_move']}")
    revisit_penalty = 0
    if new_pos in history:
        revisit_penalty = REVISIT_PENALTY  # tune
    details['R_revisit'] = -W_REVISIT * revisit_penalty

    # print(f"Revisit Penalty: {revisit_penalty}, R revisit: {details['R_revisit']}")

    # directional progress
    prev_d = abs(prev_pos[0] - dst_pos[0]) + abs(prev_pos[1] - dst_pos[1])
    new_d = abs(new_pos[0] - dst_pos[0]) + abs(new_pos[1] - dst_pos[1])
    progress = prev_d - new_d
    max_grid_dist = (GRID_W - 1) + (GRID_H - 1)

    phi_p = - prev_d
    phi_n = - new_d

    shaping = DISCOUNT * phi_n - phi_p
    # if progress > 0:
    #     progress *= 3

    details['R_dir'] = W_DIR * (progress + shaping)

    # print(f"Progress: {progress}, R Dir: {details['R_dir']}, R Move: {details['R_move']}, Total: {details['R_dir'] + details['R_move']}")

    details['R_busy_link'] = -W_NEIGH_CONG * BUSY_LINK_PENALTY if blocked['busy_link'] else 0.0
    details['R_no_vc'] = -W_NEIGH_CONG * BUSY_LINK_PENALTY if blocked['no_vc'] else 0.0
    details['R_oob'] = - W_FAULT * FAULT_PENALTY if blocked['oob'] else 0.0

    # print(f"Busy Link: {details['R_busy_link']}, No VC: {details['R_no_vc']}, OOB: {details['R_oob']}")
    details["R_cong"] = -W_FAULT * FAULT_PENALTY if (blocked['sustained'] or cong_level == 3) else 0.0
    # details["R_cong"] = 0.0
    # fault penalty
    details['R_fault'] = -W_FAULT * FAULT_PENALTY if (blocked['faulty']) else 0.0
    # print(f"Cong: {details['R_cong']}, Fault: {details['R_fault']}")

    # neighbourhood congestion
    neigh_score, neigh_fault_score, neigh_info = env_obj.neighbourhood_congestion(new_pos, active_packets=active_packets, radius=1)
    details.update({'neigh_hist': neigh_info['hist'], 'neigh_active': neigh_info['active'], 'neigh_fault': neigh_info['fault'], 'neigh_active_count': neigh_info['active_in_neigh']})
    # details['R_neigh_cong'] = -W_NEIGH_CONG * neigh_score
    details['R_neigh_cong'] = 0.0
    details['R_neigh_fault'] = -W_NEIGH_FAULT * neigh_fault_score

    # print(f"Neigh Fault: {neigh_fault_score}")
    # print("R_neigh", details['R_neigh'])
    # print("R_neigh_fault", details['R_neigh_fault'])

    age_pen = age_penalty(age)

    if age > PACKET_MAX_AGE:
        age_pen *= 5
    details['R_age'] = W_AGE * age_pen * DROP_PENALTY
    # print(f"Age: {age}, Pen: {age_pen}, R_Age: {details['R_age']}")
    # dest reward
    details['R_dest'] = W_DEL * DEST_REWARD if new_pos == dst_pos else 0.0

    # print("R_dest", details['R_dest'])

    total = sum(details[k] for k in ['R_move','R_dir', 'R_revisit', 'R_fault','R_neigh_cong','R_dest','R_neigh_fault', 'R_busy_link', 'R_oob','R_cong','R_no_vc'])
    total = float(np.clip(total, -R_MAX, R_MAX))
    details['total'] = total
    details['prev_dist'] = prev_d
    details['new_dist'] = new_d
    details['progress'] = progress
    details['cong_level'] = cong_level

    # print("R_total", details)
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

    # fig, ax = plt.subplots(figsize=(6, 6))
    # plt.ion()
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
    env.generate_random_faults(max_node_faults=MAX_FAULTS, max_link_faults=0, avoid_nodes=avoid_nodes)

    # print("Fault", env.faulty_nodes)

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
            # env.update_node_status(env.NODES)
            if pkt.done:
                if pkt in active_packets:
                    active_packets.remove(pkt)
                continue

            prev = pkt.pos
            dst_coord = env.idx_to_coord(pkt.dst_idx)

            # compute cong & dir at current node (features)
            neighbor_cong = env.compute_cong_dir_at(prev, active_packets=active_packets)
            faulty_nodes = env.faulty_at_start(prev, neighbor_cong)
            # print("Current Node:", prev,"Congestion", neighbor_cong)
            # print("faulty", faulty_nodes)

            last_node = pkt.history[-2] if len(pkt.history) > 1 else None
            # build state index
            s_idx = state_index(env.coord_to_idx(prev[0], prev[1]), pkt.dst_idx, last_node, faulty_nodes, N=GRID_W)

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

            # remove aged packets
            if pkt.age > PACKET_MAX_AGE:
                # release VC at current position
                env.release_vc_at(pkt.pos)
                if pkt in active_packets:
                    active_packets.remove(pkt)
                dropped += 1
                packet_age += pkt.age

            # compute reward
            r, details = compute_reward(prev, new_pos, dst_coord, blocked, neighbor_cong[action], env, active_packets, pkt.history, pkt.age)

            pkt.cum_reward += r
            episode_reward += r
            pkt.history.append(new_pos)
            pkt.age += 1

            # next state
            neighbor_cong = env.compute_cong_dir_at(new_pos, active_packets=active_packets)
            faulty_nodes = env.faulty_at_start(new_pos, neighbor_cong)
            s2_idx = state_index(env.coord_to_idx(new_pos[0], new_pos[1]), pkt.dst_idx, prev, faulty_nodes, N=GRID_W)

            # Q-learning update
            max_future = np.max(q_table[s2_idx])
            current_q = q_table[s_idx, action]
            if pkt.done:
                target = DEST_REWARD
            else:
                target = r + DISCOUNT * max_future
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * target
            q_table[s_idx, action] = new_q

        # draw
        # draw_grid(ax, env, active_packets, show_ports=True)
        # plt.pause(1)
        # end of timestep housekeeping
        env.end_timestep()
        episode_reward = episode_reward/max(1, injected)
    for packet in active_packets:
        packet_age += packet.age

    # if render:
    #     plt.ioff()
    #     plt.show()
    # epsilon decay
    if episode > 10 and EPSILON > EPS_MIN:
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

    if episode >= 20000 and episode % 5000 == 0:
        LEARNING_RATE = LEARNING_RATE * 0.5
    # if episode >= 5000 and episode % 5000 == 0:
    #     if INJECTION_RATE <= 0.2:
    #         INJECTION_RATE += 0.01
    # if episode >= 10000 and episode % 5000 == 0:
    #     if MAX_FAULTS < NUM_NODES/2:
    #         MAX_FAULTS += 1

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
    if episode >= 20000 and episode % 5000 == 0:
        with open(f"qtable_grid{GRID_W}x{GRID_H}_vcs{NUM_VCS}_{int(time.time())}.pkl", "wb") as f:
            pickle.dump(q_table, f)

        print("Training finished. Q-table saved.")
