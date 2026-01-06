# inference.py
import argparse
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon

# from train import NUM_NODES
from new_environment import NetworkEnv, Packet
# from train import blocked

# -------------------------
# Environment / constants - must match training
# -------------------------
GRID_W = GRID_H = 5
NUM_VCS = 4
NUM_ACTIONS = 5
NUM_NODES = GRID_H * GRID_W
MAX_TIMESTEPS = 400

MAX_ACTIVE_PACKETS = 1
PACKET_INJECTION_PROB = 1
PACKET_MAX_AGE = 40

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
W_LOCAL_CONG = 0.0
W_NEIGH_CONG = 0.2
W_NEIGH_FAULT = 0.0
W_REVISIT = 0.4
W_AGE = 0.5
W_DEL = 2
MAX_NODE_VISIT_CAP = 50.0
MAX_LINK_VISIT_CAP = 50.0
R_MAX = 1000.0


# instantiate environment
env = NetworkEnv(grid_w=GRID_W, grid_h=GRID_H, num_vcs=NUM_VCS,
                 vc_fault_persistence=1, vc_clear_persistence=1, vc_ema_alpha=0.25,
                 history_scale=50.0)


# -------------------------
# drawing helpers
# -------------------------
NODE_BG = "#efefef"
NODE_FAULTY_BG = "#d9534f"   # red-ish
NODE_SUSTAIN_BG = "#f0ad4e"  # orange
NODE_BORDER = "#444444"
PACKET_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

PORT_MARKER_SIZE = 0.12  # fraction of cell

def draw_grid(ax, env, active_packets, show_ports=True):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(0, env.GRID_W)
    ax.set_ylim(0, env.GRID_H)
    ax.invert_yaxis()  # so (0,0) is top-left like your coordinate system
    ax.set_xticks([])
    ax.set_yticks([])

    # draw cells
    for r in range(env.GRID_H):
        for c in range(env.GRID_W):
            node = (r, c)
            # base color
            face = NODE_BG
            if node in env.faulty_nodes:
                face = NODE_FAULTY_BG
            # if any sustained port exists at this node, slightly orange tinted
            node_has_sustained = any(((node, p) in env.sustained_faulty_ports) for p in [0,1,2,3,env.LOCAL_PORT])
            if node_has_sustained and (node not in env.faulty_nodes):
                face = NODE_SUSTAIN_BG

            rect = Rectangle((c, r), 1, 1, facecolor=face, edgecolor=NODE_BORDER)
            ax.add_patch(rect)

    # show port markers (triangles) for sustained ports
    if show_ports:
        for ((nr, nc), port) in env.sustained_faulty_ports:
            if not env.in_bounds(nr, nc):
                continue
            # compute triangle position inside cell according to port: 0=UP,1=RIGHT,2=DOWN,3=LEFT
            cx = nc + 0.5
            cy = nr + 0.5
            offset = 0.35
            if port == 3:  # UP - triangle pointing up
                tri = RegularPolygon((cx, cy - offset), numVertices=3, radius=0.15, orientation=0)
            elif port == 0:  # RIGHT
                tri = RegularPolygon((cx + offset, cy), numVertices=3, radius=0.15, orientation=0.5 * np.pi)
            elif port == 1:  # DOWN
                tri = RegularPolygon((cx, cy + offset), numVertices=3, radius=0.15, orientation=np.pi)
            elif port == 2:  # LEFT
                tri = RegularPolygon((cx - offset, cy), numVertices=3, radius=0.15, orientation=1.5 * np.pi)
            else:  # local
                tri = Circle((cx, cy), 0.08)
            tri.set_facecolor("#7f0000")
            tri.set_edgecolor("k")
            ax.add_patch(tri)

    # draw packets
    for i, pkt in enumerate(active_packets):
        px, py = pkt.pos  # (row,col)
        cx = py + 0.5
        cy = px + 0.5
        color = PACKET_COLORS[i % len(PACKET_COLORS)]
        circle = Circle((cx, cy), 0.25, color=color, zorder=4)
        ax.add_patch(circle)
        ax.text(cx, cy, (pkt.dst_idx, pkt.age), color="white", weight="bold", ha="center", va="center", zorder=5, fontsize=8)

    # draw destinations as small black squares (if any active packets)
    for pkt in active_packets:
        dx, dy = env.idx_to_coord(pkt.dst_idx)
        cx = dy + 0.5
        cy = dx + 0.5
        dest_rect = Rectangle((cx - 0.12, cy - 0.12), 0.24, 0.24, facecolor="#000000", zorder=3)
        ax.add_patch(dest_rect)

    ax.set_title(f"Packets: {len(active_packets)} | Sustained ports: {len(env.sustained_faulty_ports)} | Faulty nodes: {len(env.faulty_nodes)}")
    return ax


# -------------------------
# helpers (same semantics as train.py)
# -------------------------
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

    hops = np.abs(delx) + np.abs(dely)

    if hops <= 1:
        dist = 0
    elif hops == 2:
        dist = 1
    elif hops > 2:
        dist = 2
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
    combined = dir * 3 + dist
    packed = (combined << 7) | bits

    raw_state = [combined]
    raw_state.append(neigh)
    # optional: return components for debugging
    return raw_state, packed  # or return dir, bits, packed

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
    # if progress > 0:
    #     progress *= 3
    details['R_dir'] = W_DIR * (progress)

    # print(f"Progress: {progress}, R Dir: {details['R_dir']}")

    details['R_busy_link'] = -W_NEIGH_CONG * BUSY_LINK_PENALTY if blocked['busy_link'] else 0.0
    details['R_no_vc'] = -W_NEIGH_CONG * BUSY_LINK_PENALTY if blocked['no_vc'] else 0.0
    details['R_oob'] = - W_FAULT * FAULT_PENALTY if blocked['oob'] else 0.0

    # print(f"Busy Link: {details['R_busy_link']}, No VC: {details['R_no_vc']}, OOB: {details['R_oob']}")
    # details["R_cong"] = -5.0 if (blocked['sustained'] or cong_level == 3) else 0.0
    details["R_cong"] = 0.0
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
    if age >= PACKET_MAX_AGE:
        details['R_age'] = W_AGE * DROP_PENALTY
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
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.ion()

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
                neighbor_cong = env.compute_cong_dir_at(prev, active_packets=active_packets)
                faulty_nodes = env.faulty_at_start(prev, neighbor_cong)
                # print("faulty sustained", faulty_nodes)
                # build state index
                state, s_idx = state_index(env.coord_to_idx(prev[0], prev[1]), pkt.dst_idx, faulty_nodes, N=GRID_W)

                # select action (epsilon-greedy)

                action = int(np.argmax(q_table[s_idx]))

                # attempt move and commit
                blocked, traversed_link, intended = env.attempt_and_commit_move(pkt, action)

                blocked_list = list(blocked.values())
                if any(blocked_list):
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
                r, details = compute_reward(prev, new_pos, dst_coord, blocked, neighbor_cong[action], env,
                                            active_packets, pkt.history, pkt.age)

                # print("reawrd", r)
                pkt.cum_reward += r
                episode_reward += r
                pkt.age += 1
                pkt.history.add(new_pos)


                # remove aged packets
                if pkt.age > PACKET_MAX_AGE:
                    # release VC at current position
                    env.release_vc_at(pkt.pos)
                    if pkt in active_packets:
                        active_packets.remove(pkt)
                    dropped += 1
                    packet_age += pkt.age

            # draw
            draw_grid(ax, env, active_packets, show_ports=True)
            plt.pause(1)

            # end of timestep housekeeping
            env.end_timestep()
        for packet in active_packets:
            packet_age += packet.age

        if render:
            plt.ioff()
            plt.show()
        # episode done
        stats['episodes'] += 1
        stats['injected'] += injected
        stats['delivered'] += delivered
        stats['dropped'] += dropped
        stats['total_reward'] += episode_reward
        stats['total_steps'] += episode_steps
        if hops_for_delivered:
            stats['avg_hops'].append(packet_age/injected)

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


def manual_test_run(q_table, env, src, dst, faulty_nodes=None, faulty_links=None, max_steps=200, delay=0.25, render=True):
    """
    Manually run a single scenario:
      - src, dst: either flattened indices or (row,col) tuples
      - faulty_nodes: iterable of (row,col) coordinates to override (optional)
      - faulty_links: iterable of frozenset pairs of node coords (optional)
    Returns a dict of result info (delivered, steps, blocked_frac, history).
    """
    # normalize src/dst to flattened indices

    # stats = {
    #     'episodes': 0,
    #     'delivered': 0,
    #     'injected': 0,
    #     'dropped': 0,
    #     'total_reward': 0.0,
    #     'total_steps': 0,
    #     'blocked_attempts': 0,
    #     'total_attempts': 0,
    #     'avg_hops': []
    # }

    active_packets = []
    timestep = 0
    episode_reward = 0.0
    episode_steps = 0
    injected = 0
    delivered = 0
    dropped = 0
    hops_for_delivered = []
    packet_age = 0
    blocked_attempts = 0
    total_attempts = 0


    if isinstance(src, tuple):
        src_idx = env.coord_to_idx(src[0], src[1])
    else:
        src_idx = int(src)
    if isinstance(dst, tuple):
        dst_idx = env.coord_to_idx(dst[0], dst[1])
    else:
        dst_idx = int(dst)

    # reset env
    env.reset_counters()
    if faulty_nodes is not None or faulty_links is not None:
        env.set_faults(faulty_nodes if faulty_nodes is not None else [], faulty_links if faulty_links is not None else [])
    else:
        env.set_faults(set(), set())

    # inject packet if possible
    p = env.try_inject(src_idx, dst_idx)
    if not p:
        return {'error': 'could not inject at source (VC unavailable or sustained fault)'}

    active_packets = [p]
    injected += 1
    stats = {'delivered':0, 'dropped':0, 'attempts':0, 'blocked':0, 'history':[]}

    fig = None
    ax = None
    if render:
        fig, ax = plt.subplots(figsize=(6,6))
        plt.ion()

    for step in range(max_steps):
        env.start_timestep()
        # update flags for injected node if helper exists
        if hasattr(env, "update_port_flags_for_nodes"):
            env.update_port_flags_for_nodes({env.idx_to_coord(src_idx)})

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
            # print("faulty sustained", faulty_nodes)
            # build state index
            state, s_idx = state_index(env.coord_to_idx(prev[0], prev[1]), pkt.dst_idx, faulty_nodes, N=GRID_W)

            # select action (epsilon-greedy)
            print(f"State: {state}, Q-Value: {q_table[s_idx]}")
            action = int(np.argmax(q_table[s_idx]))

            # attempt move and commit
            blocked, traversed_link, intended = env.attempt_and_commit_move(pkt, action)

            blocked_list = list(blocked.values())
            if any(blocked_list):
                blocked_attempts += 1
            total_attempts += 1

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
            r, details = compute_reward(prev, new_pos, dst_coord, blocked, neighbor_cong[action], env,
                                        active_packets, pkt.history, pkt.age)

            # print("reawrd", r)
            pkt.cum_reward += r
            episode_reward += r
            pkt.age += 1
            pkt.history.add(new_pos)

            # remove aged packets
            if pkt.age > PACKET_MAX_AGE:
                # release VC at current position
                env.release_vc_at(pkt.pos)
                if pkt in active_packets:
                    active_packets.remove(pkt)
                dropped += 1
                packet_age += pkt.age

        # draw
        if render:
            draw_grid(ax, env, active_packets, show_ports=True)
            plt.pause(delay)

        env.end_timestep()
        if len(active_packets) == 0:
            break

    if render:
        plt.ioff()
        plt.show()

    # stats['steps'] = step + 1
    # stats['blocked_frac'] = stats['blocked'] / max(1, stats['attempts'])
    # return stats



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
                     history_scale=50.0)

    faulty_nodes = [(0, 0), (1,3), (3,4), (2,2)]
    src = (0,4)
    dst = (4,1)
    res = evaluate(q_table, env, episodes=args.episodes, max_timesteps=args.max_timesteps, seed=args.seed, render=False)
    # manual_test_run(q_table, env, src=src, dst=dst, faulty_nodes=faulty_nodes, delay=1, render=True)

    print("==== Evaluation summary ====")
    for k, v in res.items():
        print(f"{k:25s}: {v}")

if __name__ == '__main__':
    main()
