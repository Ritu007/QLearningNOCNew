# # environment.py
# import random
# import uuid
# from collections import defaultdict
# import math
#
# from train import NUM_NODES
#
#
# class Packet:
#     def __init__(self, src_idx, dst_idx, idx_to_coord):
#         self.id = str(uuid.uuid4())[:8]
#         self.src_idx = int(src_idx)
#         self.dst_idx = int(dst_idx)
#         self.row, self.col = idx_to_coord(self.src_idx)  # (row, col)
#         self.age = 0
#         self.cum_reward = 0.0
#         self.done = False
#
#     @property
#     def pos(self):
#         return (self.row, self.col)
#
#     def set_pos(self, row, col):
#         self.row, self.col = row, col
#
#     def idx(self, coord_to_idx):
#         return coord_to_idx(self.row, self.col)
#
# class Node:
#     def __init__(self, idx, idx_to_coord):
#         self.idx = idx
#         self. x, self.y = idx_to_coord(self.idx)
#         self.neighbours = [-1, -1, -1, -1]
# class NetworkEnv:
#     def __init__(self,
#                  grid_w=5, grid_h=5,
#                  num_vcs=2,
#                  vc_fault_persistence=5,
#                  vc_clear_persistence=3,
#                  vc_ema_alpha=0.25,
#                  history_scale=8.0):
#         # grid
#         self.GRID_W = grid_w
#         self.GRID_H = grid_h
#         self.NUM_NODES = self.GRID_W * self.GRID_H
#
#         # VC & persistence
#         self.NUM_VCS = num_vcs
#         self.VC_FAULT_PERSISTENCE = vc_fault_persistence
#         self.VC_CLEAR_PERSISTENCE = vc_clear_persistence
#         self.VC_EMA_ALPHA = vc_ema_alpha
#
#         # occupancy/history
#         self.HISTORY_SCALE = history_scale
#
#         # action mapping: (dr, dc)
#         self.DELTAS = {
#
#             0: (0, 1),   # RIGHT
#             1: (1, 0),   # DOWN
#             2: (0, -1),   # LEFT
#             3: (-1, 0)  # UP
#         }
#         self.PORTS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
#         self.NUM_ACTIONS = 4
#
#         # link storage
#         self.ALL_LINKS = set()
#         for r in range(self.GRID_H):
#             for c in range(self.GRID_W):
#                 a = (r, c)
#                 for dr, dc in self.DELTAS.values():
#                     b = (r + dr, c + dc)
#                     if self.in_bounds(b[0], b[1]):
#                         self.ALL_LINKS.add(frozenset({a, b}))
#         self.ALL_LINKS = list(self.ALL_LINKS)
#
#         self.NODES = set()
#
#         for node in range(NUM_NODES):
#             self.NODES.add(Node(node, self.idx_to_coord))
#         # runtime counters / state
#         self.reset_counters()
#
#         # persistent fault lists (set by user or generated)
#         self.faulty_nodes = set()
#         self.faulty_links = set()
#
#     # -----------------------
#     # index/coord helpers
#     # -----------------------
#     def idx_to_coord(self, idx):
#         """Return (row, col) for a flattened idx (row-major)"""
#         row = int(idx // self.GRID_W)
#         col = int(idx % self.GRID_W)
#         return row, col
#
#     def coord_to_idx(self, row, col):
#         return int(row * self.GRID_W + col)
#
#     def in_bounds(self, row, col):
#         return 0 <= row < self.GRID_H and 0 <= col < self.GRID_W
#
#     # -----------------------
#     # reset / counters
#     # -----------------------
#     def reset_counters(self):
#         # free VCs per router (persistent across timesteps unless explicitly reset)
#         self.vc_free = {}
#         for r in range(self.GRID_H):
#             for c in range(self.GRID_W):
#                 self.vc_free[(r, c)] = self.NUM_VCS
#
#         # per-timestep reservation
#         self.vc_reserved = defaultdict(int)
#
#         # sustain counters + EMA + sustained faulty set
#         self.vc_full_counter = defaultdict(int)
#         self.vc_recover_counter = defaultdict(int)
#         self.vc_cong_ema = defaultdict(float)
#         self.sustained_faulty = set()
#
#         # visited counters (for congestion)
#         self.node_visit_count = defaultdict(float)
#         self.link_visit_count = defaultdict(float)
#
#         # occupied links in a timestep
#         self.occupied_links = set()
#
#     # -----------------------
#     # faults
#     # -----------------------
#     def set_faults(self, faulty_nodes=None, faulty_links=None):
#         self.faulty_nodes = set(faulty_nodes) if faulty_nodes else set()
#         self.faulty_links = set(faulty_links) if faulty_links else set()
#
#     def generate_random_faults(self, max_node_faults=2, max_link_faults=0, avoid_nodes=None):
#         all_nodes = [(node.x, node.y) for node in self.NODES]
#         if avoid_nodes:
#             all_nodes = [n for n in all_nodes if n not in avoid_nodes]
#         node_fault_count = random.randint(0, max_node_faults)
#         faulty_nodes = set(random.sample(all_nodes, k=node_fault_count)) if node_fault_count > 0 else set()
#
#         possible_links = [L for L in self.ALL_LINKS if not (avoid_nodes and any(n in avoid_nodes for n in L))]
#         link_fault_count = random.randint(0, max_link_faults)
#         faulty_links = set(random.sample(possible_links, k=link_fault_count)) if link_fault_count > 0 else set()
#
#         self.set_faults(faulty_nodes, faulty_links)
#         return self.faulty_nodes, self.faulty_links
#
#     # -----------------------
#     # timestep lifecycle
#     # -----------------------
#     def start_timestep(self):
#         """Call at beginning of each timestep to reset per-timestep structures and update EMA/persistence."""
#         # clear per-timestep reservations and occupied links
#         self.vc_reserved.clear()
#         self.occupied_links.clear()
#
#         # update vc_cong_ema and sustained-fault counters for *all* routers
#         for r in range(self.GRID_H):
#             for c in range(self.GRID_W):
#                 node = (r, c)
#                 avail = self.vc_free.get(node, 0)
#                 occ_frac = float(max(0, self.NUM_VCS - avail)) / float(max(1, self.NUM_VCS))
#                 prev = self.vc_cong_ema.get(node, 0.0)
#                 self.vc_cong_ema[node] = (1 - self.VC_EMA_ALPHA) * prev + self.VC_EMA_ALPHA * occ_frac
#
#                 # update counters
#                 if avail <= 0:
#                     self.vc_full_counter[node] = self.vc_full_counter.get(node, 0) + 1
#                     self.vc_recover_counter[node] = 0
#                 else:
#                     self.vc_recover_counter[node] = self.vc_recover_counter.get(node, 0) + 1
#                     self.vc_full_counter[node] = 0
#
#                 # sustained fault logic
#                 if self.vc_full_counter[node] >= self.VC_FAULT_PERSISTENCE:
#                     self.sustained_faulty.add(node)
#                 if node in self.sustained_faulty and self.vc_recover_counter[node] >= self.VC_CLEAR_PERSISTENCE:
#                     self.sustained_faulty.discard(node)
#
#     def end_timestep(self):
#         """Optional cleanup at timestep end. Currently nothing required."""
#         pass
#
#     # -----------------------
#     # injection & VC handling
#     # -----------------------
#     def try_inject(self, src_idx, dst_idx):
#         """Try to inject a packet at source index; only if a VC is available and source not sustained faulty.
#            Returns Packet instance or None if injection blocked."""
#         sr, sc = self.idx_to_coord(src_idx)
#         node = (sr, sc)
#         if node in self.sustained_faulty or node in self.faulty_nodes:
#             return None
#         available = self.vc_free.get(node, 0) - self.vc_reserved.get(node, 0)
#         if available <= 0:
#             return None
#         # reserve and consume a VC immediately for injection
#         self.vc_reserved[node] += 1
#         self.vc_free[node] -= 1
#         # reservation consumed
#         self.vc_reserved[node] -= 1
#         p = Packet(src_idx, dst_idx, self.idx_to_coord)
#         return p
#
#     # -----------------------
#     # move / reservation / commit
#     # -----------------------
#     def attempt_and_commit_move(self, pkt, action):
#         """
#         Attempt to move 'pkt' using 'action'. This:
#          - checks bounds,
#          - checks sustained faults & VC availability for intended neighbour,
#          - reserves VC at dest,
#          - checks link contention, commits move if allowed (consumes dest VC, releases src VC),
#          - if blocked releases reservation and reverts packet position.
#         Returns (was_blocked:bool, traversed_link or None, intended_pos)
#         """
#         prev = pkt.pos
#         dr, dc = self.DELTAS[action]
#         intended = (prev[0] + dr, prev[1] + dc)
#
#         # out of bounds => blocked
#         if not self.in_bounds(*intended):
#             return True, None, prev
#
#         # if intended is explicitly faulty or sustained faulty => blocked
#         if intended in self.faulty_nodes or intended in self.sustained_faulty:
#             return True, None, prev
#
#         # check VC availability (consider existing reservations)
#         avail = self.vc_free.get(intended, 0) - self.vc_reserved.get(intended, 0)
#         if avail <= 0:
#             # no VC -> blocked
#             return True, None, prev
#
#         # tentatively reserve VC for intended
#         self.vc_reserved[intended] += 1
#
#         # perform move on packet object (tentative)
#         pkt.set_pos(intended[0], intended[1])
#         traversed_link = frozenset({prev, intended}) if prev != intended else None
#
#         # check link contention (occupied_links)
#         if traversed_link is not None and traversed_link in self.occupied_links:
#             # revert and release reservation
#             pkt.set_pos(prev[0], prev[1])
#             self.vc_reserved[intended] -= 1
#             return True, None, prev
#
#         # commit: consume reserved VC at dest and release a VC at source
#         self.vc_reserved[intended] -= 1
#         # decrement free at dest (reservation consumed) - but we've not consumed twice; keep consistent
#         # (vc_free already unaffected by reservation except when injection consumed earlier)
#         self.vc_free[intended] = max(0, self.vc_free.get(intended, 0) - 1)
#         # release src VC back (packet left src)
#         self.vc_free[prev] = self.vc_free.get(prev, 0) + 1
#
#         # mark link occupied and update counts
#         if traversed_link is not None:
#             self.occupied_links.add(traversed_link)
#             self.link_visit_count[traversed_link] = self.link_visit_count.get(traversed_link, 0.0) + 1.0
#         self.node_visit_count[intended] = self.node_visit_count.get(intended, 0.0) + 1.0
#
#         return False, traversed_link, intended
#
#     # -----------------------
#     # release on delivery / removal
#     # -----------------------
#     def release_vc_at(self, pos):
#         """Release one VC at router pos (row,col) when a packet departs the network or is dropped."""
#         if pos is None:
#             return
#         self.vc_free[pos] = self.vc_free.get(pos, 0) + 1
#
#     # -----------------------
#     # congestion/fault computation for a node
#     # -----------------------
#     def compute_cong_dir_at(self, node_pos, active_packets=None):
#         """
#         Return (cong_level, dir_idx).
#         cong: 0 low, 1 med, 2 high, 3 sustained/fault
#         dir: 0=UP,1=RIGHT,2=DOWN,3=LEFT (dominant problematic direction)
#         """
#         x, y = node_pos
#
#         # If current node explicitly faulty or sustained faulty or no VCs -> cong=3
#         if (x, y) in self.faulty_nodes or (x, y) in self.sustained_faulty or self.vc_free.get((x, y), 0) <= 0:
#             # pick a valid port as dir (first in order)
#             for i in range(self.NUM_ACTIONS):
#                 dr, dc = self.DELTAS[i]
#                 nx, ny = x + dr, y + dc
#                 if self.in_bounds(nx, ny):
#                     return 3, i
#             return 3, 0
#
#         occ = {}
#         fault = {}
#         for i in range(self.NUM_ACTIONS):
#             dr, dc = self.DELTAS[i]
#             nx, ny = x + dr, y + dc
#             if not self.in_bounds(nx, ny):
#                 occ[i] = 1.0
#                 fault[i] = True
#                 continue
#             neigh = (nx, ny)
#             occ_val = min(1.0, self.node_visit_count.get(neigh, 0) / max(1.0, self.HISTORY_SCALE))
#             occ[i] = occ_val
#             link = frozenset({(x, y), neigh})
#             avail_vc = self.vc_free.get(neigh, 0) - self.vc_reserved.get(neigh, 0)
#             sustained = neigh in self.sustained_faulty
#             fault[i] = (neigh in self.faulty_nodes) or (link in self.faulty_links) or sustained or (avail_vc <= 0)
#
#         # if any outgoing port faulty -> cong=3 and dir = first faulty port (priority)
#         for i in range(self.NUM_ACTIONS):
#             if fault[i]:
#                 return 3, i
#
#         # otherwise use EMA occupancy at current node to compute cong levels
#         occ_ema = self.vc_cong_ema.get((x, y), 0.0)
#         if occ_ema >= 0.66:
#             cong = 2
#         elif occ_ema >= 0.33:
#             cong = 1
#         else:
#             cong = 0
#
#         # dir: the port with maximum occupancy (tie-break by index)
#         dir_idx = max(range(self.NUM_ACTIONS), key=lambda a: (occ[a], -a))
#         return cong, dir_idx
#
#     # -----------------------
#     # neighbourhood congestion for reward shaping
#     # -----------------------
#     def neighbourhood_congestion(self, center_pos, active_packets=None, radius=1):
#         neigh_nodes = []
#         cx, cy = center_pos
#         for r in range(self.GRID_H):
#             for c in range(self.GRID_W):
#                 if abs(cx - r) + abs(cy - c) <= radius:
#                     neigh_nodes.append((r, c))
#         node_area = max(1, len(neigh_nodes))
#         total_hist = sum(self.node_visit_count.get(n, 0) for n in neigh_nodes)
#         avg_hist = total_hist / node_area
#         hist_score = float(math.tanh(avg_hist / max(1.0, self.HISTORY_SCALE)))
#
#         active_in_neigh = 0
#         if active_packets is not None:
#             for p in active_packets:
#                 if abs(p.pos[0] - cx) + abs(p.pos[1] - cy) <= radius:
#                     active_in_neigh += 1
#
#         active_scale = max(1.0, node_area * 0.5)
#         active_score = float(math.tanh(active_in_neigh / active_scale))
#
#         W_HIST, W_ACTIVE = 0.6, 0.4
#         combined = W_HIST * hist_score + W_ACTIVE * active_score
#         return float(min(max(combined, 0.0), 1.0)), {'hist': hist_score, 'active': active_score, 'active_in_neigh': active_in_neigh}
#
#

# environment.py
import random
import uuid
from collections import defaultdict
import math

class Packet:
    def __init__(self, src_idx, dst_idx, idx_to_coord):
        self.id = str(uuid.uuid4())[:8]
        self.src_idx = int(src_idx)
        self.dst_idx = int(dst_idx)
        self.row, self.col = idx_to_coord(self.src_idx)  # (row, col)
        self.age = 0
        self.cum_reward = 0.0
        self.done = False
        # pkt.vc holds the (node, port) where this packet currently occupies a VC
        # on injection we'll set it to (src_node, LOCAL_PORT)
        self.vc = None

    @property
    def pos(self):
        return (self.row, self.col)

    def set_pos(self, row, col):
        self.row, self.col = row, col

    def idx(self, coord_to_idx):
        return coord_to_idx(self.row, self.col)

class Node:
    def __init__(self, idx, idx_to_coord):
        self.idx = idx
        self. x, self.y = idx_to_coord(self.idx)
        self.neighbours = [-1, -1, -1, -1]

class NetworkEnv:
    """
    Grid network environment with per-(node,port) VC modeling.

    Port convention (for input ports at a node):
       0 = input from UP (i.e., neighbor at row-1,col)
       1 = input from RIGHT (neighbor at row, col+1)
       2 = input from DOWN (neighbor at row+1, col)
       3 = input from LEFT (neighbor at row, col-1)
       4 = LOCAL (local injection / ejection VC pool)
    When moving from node A with action a, opponent in_port on neighbor B is opp = (a+2)%4.
    """
    LOCAL_PORT = 4

    def __init__(self,
                 grid_w=5, grid_h=5,
                 num_vcs=2,
                 vc_fault_persistence=5,
                 vc_clear_persistence=3,
                 vc_ema_alpha=0.25,
                 history_scale=8.0,
                 no_vc_threshold=10,
                 sustained_congestion_threshold=30):
        # grid
        self.GRID_W = grid_w
        self.GRID_H = grid_h
        self.NUM_NODES = self.GRID_W * self.GRID_H

        # VC & persistence
        self.NUM_VCS = num_vcs
        self.VC_FAULT_PERSISTENCE = vc_fault_persistence
        self.VC_CLEAR_PERSISTENCE = vc_clear_persistence
        self.VC_EMA_ALPHA = vc_ema_alpha

        # additional congestion persistence tuning
        self.NO_VC_THRESHOLD = no_vc_threshold
        self.SUSTAINED_CONGESTION_THRESHOLD = sustained_congestion_threshold

        # occupancy/history
        self.HISTORY_SCALE = history_scale

        # action mapping: (dr, dc)
        # action a = direction to move from current node
        self.DELTAS = {

            0: (0, 1),   # RIGHT
            1: (1, 0),   # DOWN
            2: (0, -1),   # LEFT
            3: (-1, 0)  # UP
        }
        self.PORTS = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        self.NUM_ACTIONS = 4
        self.ALL_PORT_INDICES = [0,1,2,3, self.LOCAL_PORT]

        # precompute links
        self.ALL_LINKS = set()
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                a = (r, c)
                for dr, dc in self.DELTAS.values():
                    b = (r + dr, c + dc)
                    if self.in_bounds(b[0], b[1]):
                        self.ALL_LINKS.add(frozenset({a, b}))
        self.ALL_LINKS = list(self.ALL_LINKS)

        self.NODES = set()

        for node in range(self.NUM_NODES):
            self.NODES.add(Node(node, self.idx_to_coord))

        # runtime counters / state
        self.reset_counters()

        # persistent fault lists (explicitly set or generated)
        self.faulty_nodes = set()
        self.faulty_links = set()

    # -----------------------
    # index/coord helpers
    # -----------------------
    def idx_to_coord(self, idx):
        """Return (row, col) for a flattened idx (row-major)"""
        row = int(idx // self.GRID_W)
        col = int(idx % self.GRID_W)
        return row, col

    def coord_to_idx(self, row, col):
        return int(row * self.GRID_W + col)

    def in_bounds(self, row, col):
        return 0 <= row < self.GRID_H and 0 <= col < self.GRID_W

    # -----------------------
    # reset / counters (per-episode)
    # -----------------------
    def reset_counters(self):
        # free VCs per (node,port)
        # valid ports for a node: LOCAL port always valid; input ports only valid if neighbor exists
        self.vc_free = {}
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                node = (r, c)
                # local port
                self.vc_free[(node, self.LOCAL_PORT)] = self.NUM_VCS
                # input ports 0..3
                for p in range(4):
                    dr, dc = self.DELTAS[p]
                    neighbor = (r + dr, c + dc)
                    if self.in_bounds(neighbor[0], neighbor[1]):
                        self.vc_free[(node, p)] = self.NUM_VCS
                    else:
                        # invalid port (edge) -> 0 VCs (unavailable)
                        self.vc_free[(node, p)] = 0

        # per-timestep reservations keyed by (node,port)
        self.vc_reserved = defaultdict(int)

        # persistence & EMA per (node,port)
        self.vc_no_vc_counter = defaultdict(int)        # consecutive timesteps vc_free==0
        self.vc_full_counter = defaultdict(int)         # short-term full counters
        self.vc_recover_counter = defaultdict(int)
        self.vc_cong_ema = defaultdict(float)           # ema of occupancy per (node,port)
        self.vc_congestion_history = defaultdict(int)   # long-run congestion counter per (node,port)
        # self.sustained_faulty = set()                   # set of nodes considered sustained faulty (node-level)
        # sustained_faulty_ports stores keys ((row,col), port)
        # meaning the specific input port at that node is considered sustained faulty.
        self.sustained_faulty_ports = set()

        # visited counters (for congestion history)
        self.node_visit_count = defaultdict(float)
        self.link_visit_count = defaultdict(float)

        # occupied links in a timestep
        self.occupied_links = set()

    # -----------------------
    # faults utilities
    # -----------------------
    def set_faults(self, faulty_nodes=None, faulty_links=None):
        self.faulty_nodes = set(faulty_nodes) if faulty_nodes else set()
        self.faulty_links = set(faulty_links) if faulty_links else set()

    def generate_random_faults(self, max_node_faults=2, max_link_faults=0, avoid_nodes=None):
        all_nodes = [(node.x, node.y) for node in self.NODES]
        if avoid_nodes:
            all_nodes = [n for n in all_nodes if n not in avoid_nodes]
        node_fault_count = random.randint(0, max_node_faults)
        faulty_nodes = set(random.sample(all_nodes, k=node_fault_count)) if node_fault_count > 0 else set()

        possible_links = [L for L in self.ALL_LINKS if not (avoid_nodes and any(n in avoid_nodes for n in L))]
        link_fault_count = random.randint(0, max_link_faults)
        faulty_links = set(random.sample(possible_links, k=link_fault_count)) if link_fault_count > 0 else set()

        self.set_faults(faulty_nodes, faulty_links)
        return self.faulty_nodes, self.faulty_links

    # -----------------------
    # timestep lifecycle: update per-(node,port) EMA and persistence counters
    # -----------------------
    def start_timestep(self):
        """Reset per-timestep structures and update EMA/persistence for every (node,port)."""
        self.vc_reserved.clear()
        self.occupied_links.clear()

        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                node = (r,c)
                # update for each valid port (including LOCAL)
                for port in [0,1,2,3,self.LOCAL_PORT]:
                    avail = self.vc_free.get((node, port), 0)
                    # occupancy fraction for this (node,port)
                    occ_frac = float(max(0, self.NUM_VCS - avail)) / float(max(1, self.NUM_VCS))
                    prev = self.vc_cong_ema.get((node, port), 0.0)
                    self.vc_cong_ema[(node, port)] = (1 - self.VC_EMA_ALPHA) * prev + self.VC_EMA_ALPHA * occ_frac

                    # no-vc counter (consecutive timesteps)
                    if avail <= 0:
                        self.vc_no_vc_counter[(node, port)] = self.vc_no_vc_counter.get((node, port), 0) + 1
                    else:
                        self.vc_no_vc_counter[(node, port)] = 0

                    # short-term full/recover counters
                    if avail <= 0:
                        self.vc_full_counter[(node, port)] = self.vc_full_counter.get((node, port), 0) + 1
                        self.vc_recover_counter[(node, port)] = 0
                    else:
                        self.vc_recover_counter[(node, port)] = self.vc_recover_counter.get((node, port), 0) + 1
                        self.vc_full_counter[(node, port)] = 0

                    # long-run congestion history update: increment if either no-vc or high ema
                    congested_now = (self.vc_no_vc_counter[(node, port)] >= 1) or (self.vc_cong_ema[(node, port)] >= 0.5)
                    if congested_now:
                        self.vc_congestion_history[(node, port)] = self.vc_congestion_history.get((node, port), 0) + 1
                    else:
                        # slight decay to history
                        self.vc_congestion_history[(node, port)] = max(0, self.vc_congestion_history.get((node, port), 0) - 1)

                # # Node-level sustained fault decision:
                # # If any of the node's input ports exceeded short-term fault persistence -> mark node sustained_faulty
                # node_full_any = any(self.vc_full_counter.get((node, p),0) >= self.VC_FAULT_PERSISTENCE for p in [0,1,2,3,self.LOCAL_PORT])
                # if node_full_any:
                #     self.sustained_faulty.add(node)
                # # clear node sustained_faulty only when all input ports have been healthy for VC_CLEAR_PERSISTENCE
                # node_recover_all = all(self.vc_recover_counter.get((node,p),0) >= self.VC_CLEAR_PERSISTENCE for p in [0,1,2,3,self.LOCAL_PORT])
                # if node in self.sustained_faulty and node_recover_all:
                #     self.sustained_faulty.discard(node)
                #
                # # long-run: if *any* port's congestion_history exceeds sustained threshold, force node sustained
                # node_long_run = any(self.vc_congestion_history.get((node,p),0) >= self.SUSTAINED_CONGESTION_THRESHOLD for p in [0,1,2,3,self.LOCAL_PORT])
                # if node_long_run:
                #     self.sustained_faulty.add(node)
                # -------------------------------
                # PORT-level sustained-fault logic
                # -------------------------------
                # If any port's short-term full counter >= VC_FAULT_PERSISTENCE -> mark that specific port as sustained-faulty
                for p in [0, 1, 2, 3, self.LOCAL_PORT]:
                    if self.vc_full_counter.get((node, p), 0) >= self.VC_FAULT_PERSISTENCE:
                        self.sustained_faulty_ports.add((node, p))

                # Clear a port's sustained-fault flag only when that same port has vc_recover_counter >= VC_CLEAR_PERSISTENCE
                for p in [0, 1, 2, 3, self.LOCAL_PORT]:
                    if ((node, p) in self.sustained_faulty_ports) and (
                            self.vc_recover_counter.get((node, p), 0) >= self.VC_CLEAR_PERSISTENCE):
                        self.sustained_faulty_ports.discard((node, p))

                # Long-run: if a port's congestion_history exceeds SUSTAINED_CONGESTION_THRESHOLD, force that port to sustained faulty
                for p in [0, 1, 2, 3, self.LOCAL_PORT]:
                    if self.vc_congestion_history.get((node, p), 0) >= self.SUSTAINED_CONGESTION_THRESHOLD:
                        self.sustained_faulty_ports.add((node, p))

    def end_timestep(self):
        pass

    # -----------------------
    # injection: allocate LOCAL VC at source
    # -----------------------
    def try_inject(self, src_idx, dst_idx):
        """Inject only if source LOCAL VC available and node not sustained faulty."""
        sr, sc = self.idx_to_coord(src_idx)
        node = (sr, sc)
        # if node in self.sustained_faulty or node in self.faulty_nodes:
        #     return None
        # block injection if node-level explicit fault or local port is sustained-faulty
        if node in self.faulty_nodes or ((node, self.LOCAL_PORT) in self.sustained_faulty_ports):
            return None

        # check LOCAL port availability
        available = self.vc_free.get((node, self.LOCAL_PORT), 0) - self.vc_reserved.get((node, self.LOCAL_PORT), 0)
        if available <= 0:
            return None
        # reserve and consume LOCAL VC
        self.vc_reserved[(node, self.LOCAL_PORT)] += 1
        self.vc_free[(node, self.LOCAL_PORT)] = max(0, self.vc_free.get((node, self.LOCAL_PORT), 0) - 1)
        self.vc_reserved[(node, self.LOCAL_PORT)] -= 1

        p = Packet(src_idx, dst_idx, self.idx_to_coord)
        # mark packet as occupying source's LOCAL VC
        p.vc = (node, self.LOCAL_PORT)
        return p

    # -----------------------
    # movement: check neighbor's *input* VC availability (opp port) and commit
    # -----------------------
    def attempt_and_commit_move(self, pkt, action):
        """
        Attempt move pkt via action (0..3).
        Reserve neighbor's incoming VC (opp port) and commit if no contention.
        Maintain per-(node,port) vc_reserved and vc_free.
        pkt.vc holds (node,port) that the pkt currently occupies; release it on successful move.
        """
        prev = pkt.pos
        dr, dc = self.DELTAS[action]
        intended = (prev[0] + dr, prev[1] + dc)

        # bounds check
        if not self.in_bounds(*intended):
            return True, None, prev

        # explicit node faults or sustained faults block
        # if intended in self.faulty_nodes or intended in self.sustained_faulty:
        #     return True, None, prev

        # check explicit node-level faults
        if intended in self.faulty_nodes:
            return True, None, prev

        # neighbor input port that must be acquired: opp = (action + 2) % 4
        opp_port = (action + 2) % 4
        # If that specific (neighbor, opp_port) was flagged sustained-faulty, block
        if ((intended, opp_port) in self.sustained_faulty_ports):
            return True, None, prev

        # neighbor input port that must be acquired: opp = (action + 2) % 4
        opp_port = (action + 2) % 4
        # availability at neighbor's input port
        avail = self.vc_free.get((intended, opp_port), 0) - self.vc_reserved.get((intended, opp_port), 0)
        if avail <= 0:
            # blocked due to no VC on neighbor in_port
            return True, None, prev

        # tentatively reserve neighbor's input VC
        self.vc_reserved[(intended, opp_port)] += 1

        # perform tentative move on packet
        pkt.set_pos(intended[0], intended[1])
        traversed_link = frozenset({prev, intended}) if prev != intended else None

        # link contention check
        if traversed_link is not None and traversed_link in self.occupied_links:
            # revert and release reservation
            pkt.set_pos(prev[0], prev[1])
            self.vc_reserved[(intended, opp_port)] -= 1
            return True, None, prev

        # commit the move:
        # consume the neighbor's input VC (reservation -> consumed)
        self.vc_reserved[(intended, opp_port)] -= 1
        self.vc_free[(intended, opp_port)] = max(0, self.vc_free.get((intended, opp_port), 0) - 1)

        # release the VC currently held by pkt (source side)
        if pkt.vc is not None:
            src_vc_node, src_vc_port = pkt.vc
            # releasing: increment free at the source port
            self.vc_free[(src_vc_node, src_vc_port)] = self.vc_free.get((src_vc_node, src_vc_port), 0) + 1

        # now pkt occupies the neighbor's input VC
        pkt.vc = (intended, opp_port)

        # bookkeeping: occupied link & visit counts
        if traversed_link is not None:
            self.occupied_links.add(traversed_link)
            self.link_visit_count[traversed_link] = self.link_visit_count.get(traversed_link, 0.0) + 1.0
        self.node_visit_count[intended] = self.node_visit_count.get(intended, 0.0) + 1.0

        return False, traversed_link, intended

    # -----------------------
    # release on delivery / forced removal
    # -----------------------
    def release_vc_at(self, pos_and_port):
        """Release VC at (node,port) or if given only node, release LOCAL port."""
        if pos_and_port is None:
            return
        # if user passes a tuple (node,port) release that, else assume pos_and_port is node and release LOCAL
        if isinstance(pos_and_port, tuple) and len(pos_and_port) == 2 and isinstance(pos_and_port[1], int):
            node, port = pos_and_port
            self.vc_free[(node, port)] = self.vc_free.get((node, port), 0) + 1
        else:
            # assume node given
            node = pos_and_port
            self.vc_free[(node, self.LOCAL_PORT)] = self.vc_free.get((node, self.LOCAL_PORT), 0) + 1

    # -----------------------
    # compute congestion & dir; return neighbor congestion levels per direction
    # -----------------------
    def compute_cong_dir_at(self, node_pos, active_packets=None):
        """
        Returns (cong_level, dir_idx, neighbor_congs)
        neighbor_congs: dict mapping action -> cong_level (0..3) or None if out-of-bounds
        Congestion for a neighbor is computed by checking that neighbor's **input port**
        corresponding to incoming flits from current node (opp port).
        - If neighbor's vc_no_vc_counter[(neighbor, in_port)] >= NO_VC_THRESHOLD => level 3
        - If neighbor's vc_congestion_history[(neighbor, in_port)] >= SUSTAINED_CONGESTION_THRESHOLD => level 3
        - Else map per-(neighbor,in_port) EMA to 0..2 using thresholds (0.33, 0.66)
        """
        x, y = node_pos
        neighbor_congs = {}

        # compute neighbor congestion per outgoing action
        for a in range(self.NUM_ACTIONS):
            dr, dc = self.DELTAS[a]
            nx, ny = x + dr, y + dc
            if not self.in_bounds(nx, ny):
                neighbor_congs[a] = None
                continue
            neigh = (nx, ny)
            # the input port at neighbor that receives from current node is opp = (a+2)%4
            opp_port = (a + 2) % 4

            # rule 1: if neighbor input port had NO_VC_THRESHOLD consecutive no-vc timesteps -> level 3
            if self.vc_no_vc_counter.get((neigh, opp_port), 0) >= self.NO_VC_THRESHOLD:
                neighbor_congs[a] = 3
                continue

            # rule 2: if long-run congestion history for that (neigh,opp) exceeded sustained threshold -> level 3
            if self.vc_congestion_history.get((neigh, opp_port), 0) >= self.SUSTAINED_CONGESTION_THRESHOLD:
                neighbor_congs[a] = 3
                continue

            # rule 3: if neighbor node itself is explicitly faulty or sustained node-level faulty -> level 3
            # if neigh in self.faulty_nodes or neigh in self.sustained_faulty:
            #     neighbor_congs[a] = 3
            #     continue
            if ((neigh, opp_port) in self.sustained_faulty_ports) or (neigh in self.faulty_nodes):
                neighbor_congs[a] = 3
                continue

            # else use EMA at (neigh,opp_port)
            ema = self.vc_cong_ema.get((neigh, opp_port), 0.0)
            if ema >= 0.66:
                neighbor_congs[a] = 2
            elif ema >= 0.33:
                neighbor_congs[a] = 1
            else:
                neighbor_congs[a] = 0

        # Decide local cong/dir for current node (use node-level rules)
        # # If node itself is forced sustained faulty or any immediate input port had long-run sustained => cong=3
        # if (x,y) in self.faulty_nodes or (x,y) in self.sustained_faulty or any(self.vc_no_vc_counter.get(((x,y),p),0) >= self.NO_VC_THRESHOLD for p in [0,1,2,3,self.LOCAL_PORT]):
        #     # choose first valid neighbor to set as dir (or 0)
        #     for i in range(self.NUM_ACTIONS):
        #         if neighbor_congs.get(i) is not None:
        #             return 3, i, neighbor_congs
        #     return 3, 0, neighbor_congs

        # If current node itself has any input port in sustained_faulty_ports or no_vc counters >= NO_VC_THRESHOLD, treat as cong=3
        if any(((x, y), p) in self.sustained_faulty_ports for p in [0, 1, 2, 3, self.LOCAL_PORT]) or any(
                self.vc_no_vc_counter.get(((x, y), p), 0) >= self.NO_VC_THRESHOLD for p in
                [0, 1, 2, 3, self.LOCAL_PORT]):
            for i in range(self.NUM_ACTIONS):
                if neighbor_congs.get(i) is not None:
                    return 3, i, neighbor_congs
            return 3, 0, neighbor_congs

        # if any neighbor has level 3, pick first such dir and set cong=3 (avoid moving into it)
        for i in range(self.NUM_ACTIONS):
            if neighbor_congs.get(i) == 3:
                return 3, i, neighbor_congs

        # otherwise use node-level EMA average across its input ports to estimate cong 0/1/2
        # compute average EMA across the valid input ports
        ema_vals = []
        for p in range(4):
            if ( (x,y), p ) in self.vc_cong_ema:
                ema_vals.append(self.vc_cong_ema.get(((x,y),p), 0.0))
            else:
                # if port invalid, skip
                ema_vals.append(0.0)
        avg_ema = sum(ema_vals) / max(1, len(ema_vals))
        if avg_ema >= 0.66:
            cong = 2
        elif avg_ema >= 0.33:
            cong = 1
        else:
            cong = 0

        # dir = port with maximum neighbor occupancy proxy (based on node_visit counts)
        occ = {}
        for i in range(self.NUM_ACTIONS):
            dr, dc = self.DELTAS[i]
            nx, ny = x + dr, y + dc
            if not self.in_bounds(nx, ny):
                occ[i] = -1.0
            else:
                neigh = (nx, ny)
                occ[i] = min(1.0, self.node_visit_count.get(neigh, 0) / max(1.0, self.HISTORY_SCALE))
        dir_idx = max(range(self.NUM_ACTIONS), key=lambda a: (occ[a], -a))
        return cong, dir_idx, neighbor_congs

    # -----------------------
    # neighbourhood congestion for reward shaping (keeps node-level)
    # -----------------------
    def neighbourhood_congestion(self, center_pos, active_packets=None, radius=1):
        neigh_nodes = []
        cx, cy = center_pos
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                if abs(cx - r) + abs(cy - c) <= radius:
                    neigh_nodes.append((r, c))
        node_area = max(1, len(neigh_nodes))
        total_hist = sum(self.node_visit_count.get(n, 0) for n in neigh_nodes)
        avg_hist = total_hist / node_area
        hist_score = float(math.tanh(avg_hist / max(1.0, self.HISTORY_SCALE)))

        active_in_neigh = 0
        if active_packets is not None:
            for p in active_packets:
                if abs(p.pos[0] - cx) + abs(p.pos[1] - cy) <= radius:
                    active_in_neigh += 1

        active_scale = max(1.0, node_area * 0.5)
        active_score = float(math.tanh(active_in_neigh / active_scale))

        W_HIST, W_ACTIVE = 0.6, 0.4
        combined = W_HIST * hist_score + W_ACTIVE * active_score
        return float(min(max(combined, 0.0), 1.0)), {'hist': hist_score, 'active': active_score, 'active_in_neigh': active_in_neigh}
