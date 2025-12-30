# inference.py
import argparse
import pickle
import random
import time
from collections import defaultdict

import numpy as np

# from train import NUM_NODES
from updated_env import NetworkEnv, Packet

# -------------------------
# Environment / constants - must match training
# -------------------------
GRID_W = GRID_H = 5
NUM_VCS = 2
NUM_ACTIONS = 4
NUM_NODES = GRID_H * GRID_W
MAX_TIMESTEPS = 400

MAX_ACTIVE_PACKETS = 10
PACKET_INJECTION_PROB = 0.25
PACKET_MAX_AGE = 200

# Reward / scaling constants - copied from train.py (keep identical)
MOVE_PENALTY = 2
DEST_REWARD = 200
W_MOVE = 2.0
W_DIR = 20.0
W_FAULT = 30.0
W_LOCAL_CONG = 2.0
W_NEIGH_CONG = 10.0
MAX_NODE_VISIT_CAP = 50.0
MAX_LINK_VISIT_CAP = 50.0
R_MAX = 1000.0

# instantiate environment
env = NetworkEnv(grid_w=GRID_W, grid_h=GRID_H, num_vcs=NUM_VCS,
                 vc_fault_persistence=5, vc_clear_persistence=3, vc_ema_alpha=0.25,
                 history_scale=8.0)

# -------------------------
# helpers (same semantics as train.py)
# -------------------------
def state_index(source, destination, faulty_routers, N=GRID_W):
    """
    Same state packing used in training.
    source, destination: flattened indices (0..NUM_NODES-1)
    faulty_routers: iterable of (r,c) tuples to be treated as 'faulty' in the state.
    """
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

    bits = 0
    for off in deltas:
        bits = (bits << 1) | is_fault((sx + off[0], sy + off[1]))
    for off in delta_th[dir]:
        bits = (bits << 1) | is_fault((sx + off[0], sy + off[1]))

    packed = (dir << 8) | bits
    return packed

def compute_reward(prev_pos, new_pos, dst_pos, was_blocked, cong_level, env_obj, active_packets):
    """
    Compose reward using env's counters and neighbourhood functions.
    """
    details = {}
    # base move
    stayed = (prev_pos == new_pos)
    base_move_cost = MOVE_PENALTY + 10 if stayed else MOVE_PENALTY
    details['R_move'] = -W_MOVE * base_move_cost

    # print("R_move", details['R_move'])

    # directional progress
    prev_d = abs(prev_pos[0] - dst_pos[0]) + abs(prev_pos[1] - dst_pos[1])
    new_d = abs(new_pos[0] - dst_pos[0]) + abs(new_pos[1] - dst_pos[1])
    progress = prev_d - new_d
    max_grid_dist = (GRID_W - 1) + (GRID_H - 1)
    details['R_dir'] = W_DIR * (progress / max(1, max_grid_dist))

    # print("R_dir", details['R_dir'])

    # fault penalty
    details['R_fault'] = -W_FAULT if (was_blocked or cong_level == 3) else 0.0

    # print("R_fault", details['R_fault'])

    # local congestion penalty
    node_count = min(env_obj.node_visit_count.get(new_pos, 0.0), MAX_NODE_VISIT_CAP)
    traversed_link = frozenset({prev_pos, new_pos}) if prev_pos != new_pos else None
    link_count = min(env_obj.link_visit_count.get(traversed_link, 0.0) if traversed_link is not None else 0.0, MAX_LINK_VISIT_CAP)
    node_norm = node_count / MAX_NODE_VISIT_CAP
    link_norm = link_count / MAX_LINK_VISIT_CAP
    details['R_local_cong'] = -W_LOCAL_CONG * (node_norm + link_norm)

    # print("R_local_cong", details['R_local_cong'])

    # neighbourhood congestion
    neigh_score, neigh_info = env_obj.neighbourhood_congestion(new_pos, active_packets=active_packets, radius=1)
    details.update({'neigh_hist': neigh_info['hist'], 'neigh_active': neigh_info['active'], 'neigh_active_count': neigh_info['active_in_neigh']})
    details['R_neigh'] = -W_NEIGH_CONG * neigh_score

    # print("R_neigh", details['R_neigh'])

    # dest reward
    details['R_dest'] = DEST_REWARD if new_pos == dst_pos else 0.0

    # print("R_dest", details['R_dest'])

    total = sum(details[k] for k in ['R_move','R_dir','R_fault','R_local_cong','R_neigh','R_dest'])
    total = float(np.clip(total, -R_MAX, R_MAX))
    details['total'] = total
    details['prev_dist'] = prev_d
    details['new_dist'] = new_d
    details['progress'] = progress
    details['cong_level'] = cong_level

    # print("R_total", details['total'])
    return total, details
# -------------------------
# Inference / evaluation loop
# -------------------------
def evaluate(q_table, env, episodes=100, max_timesteps=400, seed=None, render=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    stats = {
        'episodes': 0,
        'delivered': 0,
        'injected': 0,
        'dropped': 0,
        'total_reward': 0.0,
        'total_steps': 0,
        'blocked_attempts': 0,
        'total_attempts': 0,
        'avg_hops': []
    }

    for episode in range(episodes):
        env.reset_counters()
        active_packets = []
        timestep = 0
        episode_reward = 0.0
        episode_steps = 0
        injected = 0
        delivered = 0
        dropped = 0
        hops_for_delivered = []
        packet_age = 0

        # inject one initial packet (random) like training
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
            # print("Timestep", timestep)
            # print("fault", env.faulty_nodes)
            timestep += 1
            env.start_timestep()

            # injection attempts
            if len(active_packets) < MAX_ACTIVE_PACKETS and random.random() < PACKET_INJECTION_PROB:
                # s = random.randrange(NUM_NODES)
                # d = random.randrange(NUM_NODES)
                # while d == s:
                #     d = random.randrange(NUM_NODES)

                s = random.randrange(NUM_NODES)
                d = random.randrange(NUM_NODES)

                while env.idx_to_coord(d) in env.faulty_nodes:
                    d = random.randrange(NUM_NODES)
                while env.idx_to_coord(s) in env.faulty_nodes:
                    s = random.randrange(NUM_NODES)
                while d == s:
                    d = random.randrange(NUM_NODES)

                # active_packets.append(Packet(s, d))
                # skip inject if node explicitly faulty or sustained faulty
                sr, sc = env.idx_to_coord(s)
                if (sr, sc) in env.sustained_faulty_ports:
                    pass
                else:
                    pnew = env.try_inject(s, d)
                    if pnew:
                        injected += 1
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
                cong_level, dir_idx, neighbor_cong = env.compute_cong_dir_at(prev, active_packets=active_packets)
                faulty_nodes = env.faulty_at_start(prev, neighbor_cong)
                # print("faulty sustained", faulty_nodes)
                # build state index
                s_idx = state_index(env.coord_to_idx(prev[0], prev[1]), pkt.dst_idx, faulty_nodes, N=GRID_W)

                # select action (epsilon-greedy)

                action = int(np.argmax(q_table[s_idx]))

                # attempt move and commit
                was_blocked, traversed_link, intended = env.attempt_and_commit_move(pkt, action)

                if was_blocked:
                    stats['blocked_attempts'] += 1
                stats['total_attempts'] += 1

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
                    hops_for_delivered.append(pkt.age)

                # compute reward
                r, details = compute_reward(prev, new_pos, dst_coord, was_blocked, neighbor_cong[action], env,
                                            active_packets)

                # print("reawrd", r)
                pkt.cum_reward += r
                episode_reward += r
                pkt.age += 1

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

        # episode done
        stats['episodes'] += 1
        stats['injected'] += injected
        stats['delivered'] += delivered
        stats['dropped'] += dropped
        stats['total_reward'] += episode_reward
        stats['total_steps'] += episode_steps
        if hops_for_delivered:
            stats['avg_hops'].append(sum(hops_for_delivered) / len(hops_for_delivered))

    # Aggregate results
    delivered = stats['delivered']
    injected = stats['injected']
    dropped = stats['dropped']
    episodes = stats['episodes']
    avg_reward = stats['total_reward'] / max(1, episodes)
    delivered_rate = delivered / max(1, injected)
    avg_steps = stats['total_steps'] / max(1, episodes)
    blocked_frac = stats['blocked_attempts'] / max(1, stats['total_attempts'])
    avg_hops = np.mean(stats['avg_hops']) if stats['avg_hops'] else float('nan')

    return {
        'episodes': episodes,
        'injected': injected,
        'delivered': delivered,
        'dropped': dropped,
        'delivered_rate': delivered_rate,
        'avg_reward': avg_reward,
        'avg_steps_per_episode': avg_steps,
        'blocked_frac': blocked_frac,
        'avg_hops': avg_hops
    }

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qtable', type=str, required=True, help='Path to qtable pickle file saved from training')
    p.add_argument('--episodes', type=int, default=400, help='Number of evaluation episodes')
    p.add_argument('--max-timesteps', type=int, default=400, help='Max timesteps per episode')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    args = p.parse_args()

    # load q-table
    with open(args.qtable, 'rb') as f:
        q_table = pickle.load(f)

    print("Loaded q_table shape:", q_table.shape)

    # create env
    env = NetworkEnv(grid_w=GRID_W, grid_h=GRID_H, num_vcs=NUM_VCS,
                     vc_fault_persistence=5, vc_clear_persistence=3, vc_ema_alpha=0.25,
                     history_scale=8.0)

    res = evaluate(q_table, env, episodes=args.episodes, max_timesteps=args.max_timesteps, seed=args.seed)
    print("==== Evaluation summary ====")
    for k, v in res.items():
        print(f"{k:25s}: {v}")

if __name__ == '__main__':
    main()
