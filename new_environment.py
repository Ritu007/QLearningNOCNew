# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import random
#
# class Network:
#     def __init__(self, faulty_routers, faulty_links,  size=4):
#         self.size = size
#         self.routers = []
#         self.edges = []
#         self.n_routers = size * size
#         self.n_edges = 0
#         self.faulty_routers = faulty_routers
#         self.faulty_links = faulty_links
#         self.data = []
#         self.n_data = 1
#
#     def init_routers(self):
#         self.routers = []
#         y_coordinate = 0
#         x_coordinate = 0
#         x_max = self.size - 1
#         y_max = self.size - 1
#
#         for i in range(self.n_routers):
#             self.routers.append(Router(x_coordinate, y_max))
#             if (x_coordinate == x_max):
#                 x_coordinate = 0
#                 y_max -= 1
#             else:
#                 x_coordinate += 1
#
#     def init_edges(self):
#         # Define action order: 0=left,1=down,2=right,3=up
#         self.edges = []
#         self.n_edges = 0
#         # for r in self.routers:
#         #     # Initialize neighbor edges to -1 for each action
#         #     r.edge = [-1, -1, -1, -1]
#         #     r.neighbor = [-1, -1, -1, -1]
#
#         for i in range(self.n_routers):
#             # Compute squared distances to all other routers
#             dis = []
#             for j in range(self.n_routers):
#                 dx = self.routers[j].x - self.routers[i].x
#                 dy = self.routers[j].y - self.routers[i].y
#                 dis.append([(dx * dx + dy * dy), j, dx, dy])
#             # Sort by distance ascending
#             dis.sort(key=lambda x: x[0])
#
#             # For each candidate neighbor at exactly distance 0.0625 (0.25 actual)
#             for dist_sq, j, dx, dy in dis:
#                 # print("Dist sq", dist_sq)
#                 if i == j or dist_sq != 1:
#                     continue
#                 # Determine direction index
#                 if dx < 0 and dy == 0:
#                     direction = 0  # left
#                 elif dx == 0 and dy > 0:
#                     direction = 1  # down
#                 elif dx > 0 and dy == 0:
#                     direction = 2  # right
#                 elif dx == 0 and dy < 0:
#                     direction = 3  # up
#                 else:
#                     # not a direct cardinal neighbor
#                     continue
#
#                 # print("dire", direction)
#                 # Create edge only once, and record its index
#                 if j not in self.routers[i].neighbor:
#                     # add to neighbor lists
#                     self.routers[i].neighbor[direction] = j
#                     self.routers[j].neighbor[(direction + 2) % 4] = i
#
#                     # instantiate Edge and record index
#                     edge = Edge(min(i, j), max(i, j), math.sqrt(dist_sq))
#                     self.edges.append(edge)
#                     edge_idx = self.n_edges
#                     self.n_edges += 1
#
#                     # assign to neighbor_edges arrays
#                     self.routers[i].edge[direction] = edge_idx
#                     # compute opposite direction: (direction+2)%4
#                     opp = (direction + 2) % 4
#                     self.routers[j].edge[opp] = edge_idx
#
#     def vis_routers(self):
#         for i in range(self.n_routers):
#             print(self.routers[i].x, self.routers[i].y)
#             plt.scatter(self.routers[i].x, self.routers[i].y, color='red')
#         plt.show()
#
#     def vis_edges(self):
#         for i in range(self.n_routers):
#             plt.scatter(self.routers[i].x, self.routers[i].y, color='orange')
#         for e in self.edges:
#             plt.plot([self.routers[e.start].x, self.routers[e.end].x], [self.routers[e.start].y, self.routers[e.end].y], color='black')
#         plt.show()
#
#     def render_topology(self, pause_time=0.3):
#         """
#         routers: list of Router objects, each has .x, .y
#         edges:   list of Edge objects, each has .start, .end
#         data:    list of Data packets, each has .now, .edge
#         """
#         # print("data in render", self.data)
#         plt.clf()  # clear previous frame
#         # draw routers
#
#         for router in self.routers:
#             xs = router.x
#             ys = router.y
#             plt.scatter(xs, ys, color="lightgray", s=300, zorder=1)
#
#             if router in self.faulty_routers:
#                 xf = xs
#                 yf = ys
#                 plt.scatter(xf, yf, color="black", s=250, zorder=2)
#             plt.text(xs, ys, str(xs  + (self.size - 1 - ys) * self.size), color="red", ha="center", va="center", fontsize=8, zorder=5)
#
#
#         # xs = [r.x for r in self.routers]
#         # ys = [r.y for r in self.routers]
#         # xf = [self.routers[r].x for r in self.faulty_routers]
#         # yf = [self.routers[r].y for r in self.faulty_routers]
#         # # print(f"xf {xf}, yf {yf}")
#         # plt.scatter(xs, ys, color="lightgray", s=300, zorder=1)
#         # plt.scatter(xf, yf, color="black", s=250, zorder=2)
#         # plt.text(xs, ys, str(xs * self.size + ys), color="w", ha="center", va="center", fontsize=8, zorder=5)
#         # draw links
#         for idx, e in enumerate(self.edges):
#             x0, y0 = self.routers[e.start].x, self.routers[e.start].y
#             x1, y1 = self.routers[e.end].x, self.routers[e.end].y
#             plt.plot([x0, x1], [y0, y1], color="k", zorder=0)
#             if e.load > 0:
#                 plt.plot([x0, x1], [y0, y1], color="m", zorder=0)
#             if e.is_faulty == 1:
#                 plt.plot([x0, x1], [y0, y1], color="y", zorder=0)
#
#             mx, my = (x0 + x1) / 2, (y0 + y1) / 2
#             plt.text(mx, my, str(idx), color="red", ha="center", va="center", fontsize=8, zorder=5)
#
#
#         # draw packets
#         for i, pkt in enumerate(self.data):
#             # print(i)
#             if pkt is None:
#                 continue
#             if pkt.edge != -1:
#                 # packet is traversing an edge: draw at its midpoint
#                 e = self.edges[pkt.edge]
#                 x0, y0 = self.routers[e.start].x, self.routers[e.start].y
#                 x1, y1 = self.routers[e.end].x, self.routers[e.end].y
#                 mx, my = (x0 + x1) / 2, (y0 + y1) / 2
#             else:
#                 # packet is sitting at a router
#                 mx, my = self.routers[pkt.now].x, self.routers[pkt.now].y
#                 # Get target position
#             tx, ty = self.routers[pkt.target].x, self.routers[pkt.target].y
#
#             plt.scatter(mx, my, color="C1", s=100, zorder=4)
#             # plt.text(mx, my, str(i), color="w", ha="center", va="center", fontsize=8, zorder=5)
#
#             plt.scatter(tx, ty, color="red", s=180, zorder=3)
#             # plt.text(tx, ty, str(i), color="w", ha="center", va="center", fontsize=8, zorder=5)
#
#         plt.axis('equal')
#         plt.xticks([])
#         plt.yticks([])
#         plt.pause(pause_time)
#
#
#     def get_network_edges(self):
#         print("total no of edges: ",self.n_edges)
#         for i in range(self.n_routers):
#             print("Router {}: x = {} y = {} neighbor = {} edges = {} two_hops = {} faulty = {}".format(i, self.routers[i].x, self.routers[i].y,
#                                                                              self.routers[i].neighbor, self.routers[i].edge, self.routers[i].two_hops, self.routers[i].is_faulty))
#         for i, e in enumerate(self.edges):
#             print("Edge {}: start_point = {} end_point = {} length = {} load = {}".format(i, self.edges[i].start, self.edges[i].end,
#                                                                                 self.edges[i].len, self.edges[i].load))
#
#     def create_data(self, source, destination, priority):
#         data = Data(source, destination, np.random.random(), priority)
#         # data.start_step = self.time_steps
#         return data
#
#     def set_data(self):
#         packet_nos = self.n_data
#
#         for i in range(packet_nos):
#             destination = np.random.randint(self.n_routers)
#             while destination in self.faulty_routers:
#                 destination = np.random.randint(self.n_routers)
#
#             source = np.random.randint(self.n_routers)
#             while source in self.faulty_routers or source == destination:
#                 source = np.random.randint(self.n_routers)
#
#             self.data.append(self.create_data(source, destination, i))
#             # self.available.pop(i)
#         # Priority has been passed according to order of creation
#
#         for j in range(packet_nos, self.n_data):
#             self.data.append(None)
#
#     def get_data_packet_info(self):
#
#         for i in range(self.n_data):
#             if self.data[i] is None:
#                 print("None")
#                 continue
#             print(
#                 "Data Packet {}: start_point = {}, target  ={}, size = {}, priority = {}, edge = {}, steps = {}".format(
#                     i, self.data[i].now,
#                     self.data[i].target,
#                     self.data[i].size,
#                     self.data[i].priority,
#                     self.data[i].edge,
#                     self.data[i].total_steps))
#
#     def set_faults(self):
#         for i in range(len(self.routers)):
#             if i in self.faulty_routers:
#                 self.routers[i].is_faulty = 1
#
#         for j in range(len(self.edges)):
#             if j in self.faulty_links:
#                 self.edges[j].is_faulty = 1
#
#
#
#     def reset(self):
#         self.init_routers()
#         self.init_edges()
#         self.set_faults()
#
#         self.data = []
#         # print(self.routers)
#         # print(self.edges)
#         # self.vis_routers()
#         # self.vis_edges()
#         # self.get_network_edges()
#         # plt.ion()
#         # self.render_topology(pause_time=2)
#         # plt.ioff()
#         # plt.show()
#         self.set_data()
#         # self.get_data_packet_info()
#         obs = self.get_observation()
#
#         return obs
#
#
#     def get_observation(self):
#         obs = []
#         for data in self.data:
#             ob = (data.now, data.target)
#             obs.append(ob)
#         return obs
#
#     def step(self, action):
#         rewards = 0
#         done = False
#         for packet in self.data:
#             current_node = packet.now
#             # print("Current Node", current_node)
#             end_node = packet.target
#             if current_node == end_node:
#                 rewards += 100
#                 done = True
#                 break
#             next_node = self.routers[current_node].neighbor[action]
#             # print("next Node", next_node)
#             next_edge = self.routers[current_node].edge[action]
#             # print("next edge", next_edge)
#
#             if next_node < 0:
#                 rewards -= 20
#             elif next_node in self.faulty_routers:
#                 rewards -= 50
#             elif next_edge in self.faulty_links:
#                 rewards -= 50
#             else:
#                 packet.now = next_node
#                 rewards -= 15
#             # print("rewards", rewards)
#             # print("done", done)
#         observation = self.get_observation()
#         return observation, rewards, done
#
#
# class Router(object):
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y
#         self.neighbor = [-1,-1,-1,-1]
#         self.edge = [-1,-1,-1,-1]
#         self.is_faulty = 0
#         self.two_hops = [-1, -1, -1,-1,-1,-1,-1,-1]
#         # self.packets = []
#
#
# class Edge(object):
#     def __init__(self,x,y,l):
#         self.start = x
#         self.end = y
#         self.len = int(l+1)
#         self.load = 0
#         self.is_faulty = 0
#
#
# class Data(object):
#     def __init__(self,x,y,size,priority):
#         self.prev = None
#         self.now = x
#         self.target = y
#         self.size = size
#         self.priority = priority
#         self.time = 0
#         self.edge = -1
#         self.neigh = [priority,-1,-1,-1]
#         self.prev_dist = 0
#         self.delivered = False
#         self.is_empty = False
#         # self.start_step = -1
#         # self.end_step = -1
#         self.total_steps = 0
#
#
# # class Data:
# #     def __init__(self, x=0, y=0, inference=False, size=5):
# #         self.size = size
# #
# #         if inference:
# #             self.x = x
# #             self.y = y
# #
# #         else:
# #             self.x = np.random.randint(0, self.size)
# #             self.y = np.random.randint(0, self.size)
# #         self.move_weight = 0
# #
# #     def __str__(self):
# #         return f"{self.x}, {self.y}"
# #
# #     def __sub__(self, other):
# #         return (self.x-other.x, self.y-other.y)
# #
# #     def action(self, choice):
# #         if choice == 0:
# #             self.move(del_x=0, del_y=-1)
# #             self.move_weight = np.random.randint(0, 5)
# #         elif choice == 1:
# #             self.move(del_x=1, del_y=0)
# #             self.move_weight = np.random.randint(0, 5)
# #         elif choice == 2:
# #             self.move(del_x=0, del_y=1)
# #             self.move_weight = np.random.randint(0, 5)
# #         elif choice == 3:
# #             self.move(del_x=-1, del_y=0)
# #             self.move_weight = np.random.randint(0, 5)
# #
# #     def move(self, del_x=False, del_y=False):
# #         self.x += del_x
# #         self.y += del_y
# #
# #         if self.x < 0:
# #             self.x = 0
# #         elif self.x > self.size - 1:
# #             self.x = self.size - 1
# #
# #         if self.y < 0:
# #             self.y = 0
# #         elif self.y > self.size - 1:
# #             self.y = self.size - 1


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
        self.history = []

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
                 vc_fault_persistence=1,
                 vc_clear_persistence=1,
                 vc_ema_alpha=0.25,
                 history_scale= 100.0,
                 no_vc_threshold=1,
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

        STAY_ACTION = 4
        self.DELTAS[STAY_ACTION] = (0, 0)

        self.PORTS = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        self.NUM_ACTIONS = 5
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
        # New unified structure
        self.vc_state = defaultdict(lambda: {
            'no_vc_streak': 0,  # consecutive timesteps with avail == 0
            'recover_streak': 0,  # consecutive timesteps with avail > 0
            'ema': 0.0,  # occupancy EMA [0..1]
            'history': 0,  # long-run congestion counter (integer)
            'sustained': False  # boolean, authoritative sustained flag for this (node,port)
        })
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
        # print("faulty nodes", faulty_nodes)
        return self.faulty_nodes, self.faulty_links

    # def faulty_at_start(self,current_node, neighbor_cong):
    #     faulty_nodes = self.faulty_nodes
    #     x, y = current_node
    #
    #     for a in range(self.NUM_ACTIONS):
    #         dr, dc = self.DELTAS[a]
    #         nx, ny = x + dr, y + dc
    #         if neighbor_cong[a] is None or neighbor_cong[a] == 3:
    #             faulty_nodes.add((nx, ny))
    #             continue
    #     return faulty_nodes

    def faulty_at_start(self, current_node, neighbor_cong):
        """
        Return a new set of neighbor node coords that should be considered faulty
        for the state of `current_node` based on neighbor_cong.
        Does NOT mutate self.faulty_nodes.
        """
        # start with explicit faulty nodes (copy) or start empty if you prefer
        faulty_nodes = set(self.faulty_nodes)  # copy so we don't modify the original

        x, y = current_node
        for a in range(self.NUM_ACTIONS - 1):
            dr, dc = self.DELTAS[a]
            nx, ny = x + dr, y + dc

            # skip out-of-bounds neighbors
            if not self.in_bounds(nx, ny):
                continue

            # safe access in case neighbor_cong is a dict with missing keys
            lvl = neighbor_cong.get(a, None)

            # decide when to mark neighbor faulty for this source node
            # usually we only mark when level == 3 (sustained / no-vc)
            if lvl == 3:
                faulty_nodes.add((nx, ny))

        return faulty_nodes

    # -----------------------
    # timestep lifecycle: update per-(node,port) EMA and persistence counters
    # -----------------------
    def start_timestep(self):
        """Reset per-timestep structures and update EMA/persistence for every (node,port)."""
        self.vc_reserved.clear()
        self.occupied_links.clear()

        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                node = (r, c)
                # iterate ports (0..3) + LOCAL_PORT if used
                for port in [0, 1, 2, 3, self.LOCAL_PORT]:
                    key = (node, port)
                    avail = self.vc_free.get(key, 0)  # current free VCs (0..NUM_VCS)
                    # occupancy fraction: fraction of used VCs
                    occ_frac = float(max(0, self.NUM_VCS - avail)) / float(max(1, self.NUM_VCS))
                    # print(f"Node: {node, port}, Avail: {avail}")
                    st = self.vc_state[key]

                    # print(f"Key: {key}, State: {st}")
                    # EMA update
                    prev_ema = st['ema']
                    st['ema'] = (1 - self.VC_EMA_ALPHA) * prev_ema + self.VC_EMA_ALPHA * occ_frac

                    # streak updates
                    if avail <= 0:
                        st['no_vc_streak'] = st.get('no_vc_streak', 0) + 1
                        st['recover_streak'] = 0
                    else:
                        st['recover_streak'] = st.get('recover_streak', 0) + 1
                        st['no_vc_streak'] = 0

                    # long-run history update (increment if currently congested by EMA or no_vc)
                    congested_now = (st['no_vc_streak'] >= 1) or (st['ema'] >= 0.66)
                    if congested_now:
                        st['history'] = st.get('history', 0) + 1
                    else:
                        # gentle decay
                        st['history'] = max(0, st.get('history', 0) - 15)

                    # sustained flag logic (single place controlling policy)
                    # set sustained if either short-term full_streak meets persistence
                    if st['no_vc_streak'] >= self.VC_FAULT_PERSISTENCE:
                        st['sustained'] = True
                    # OR if long-run history exceeded threshold
                    if st['history'] >= self.SUSTAINED_CONGESTION_THRESHOLD:
                        st['sustained'] = True
                    # clear sustained only if recover streak exceeds clear persistence
                    if st['sustained'] and st['recover_streak'] >= self.VC_CLEAR_PERSISTENCE:
                        st['sustained'] = False

                    # sync set for quick lookup
                    if st['sustained']:
                        self.sustained_faulty_ports.add(key)
                    else:
                        # remove if present
                        if key in self.sustained_faulty_ports:
                            self.sustained_faulty_ports.discard(key)

    def update_node_status(self, nodes):
        """Reset per-timestep structures and update EMA/persistence for every (node,port)."""
        self.vc_reserved.clear()
        self.occupied_links.clear()

        for node in range(nodes):
            # node = (r, c)
            # iterate ports (0..3) + LOCAL_PORT if used
            for port in [0, 1, 2, 3, self.LOCAL_PORT]:
                key = (node, port)
                avail = self.vc_free.get(key, 0)  # current free VCs (0..NUM_VCS)
                # occupancy fraction: fraction of used VCs
                occ_frac = float(max(0, self.NUM_VCS - avail)) / float(max(1, self.NUM_VCS))
                print(f"Node: {node, port}, Avail: {avail}")
                st = self.vc_state[key]
                # EMA update
                prev_ema = st['ema']
                st['ema'] = (1 - self.VC_EMA_ALPHA) * prev_ema + self.VC_EMA_ALPHA * occ_frac

                # streak updates
                if avail <= 0:
                    st['no_vc_streak'] = st.get('no_vc_streak', 0) + 1
                    st['recover_streak'] = 0
                else:
                    st['recover_streak'] = st.get('recover_streak', 0) + 1
                    st['no_vc_streak'] = 0

                # long-run history update (increment if currently congested by EMA or no_vc)
                congested_now = (st['no_vc_streak'] >= 1) or (st['ema'] >= 0.66)
                if congested_now:
                    st['history'] = st.get('history', 0) + 1
                else:
                    # gentle decay
                    st['history'] = max(0, st.get('history', 0) - 15)

                # sustained flag logic (single place controlling policy)
                # set sustained if either short-term full_streak meets persistence
                if st['no_vc_streak'] >= self.VC_FAULT_PERSISTENCE:
                    st['sustained'] = True
                # OR if long-run history exceeded threshold
                if st['history'] >= self.SUSTAINED_CONGESTION_THRESHOLD:
                    st['sustained'] = True
                # clear sustained only if recover streak exceeds clear persistence
                if st['sustained'] and st['recover_streak'] >= self.VC_CLEAR_PERSISTENCE:
                    st['sustained'] = False

                # sync set for quick lookup
                if st['sustained']:
                    self.sustained_faulty_ports.add(key)
                else:
                    # remove if present
                    if key in self.sustained_faulty_ports:
                        self.sustained_faulty_ports.discard(key)

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
        p.history.append(node)
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
        blocked = {
                    "oob": False,
                    "faulty": False,
                    "sustained": False,
                    "no_vc": False,
                    "busy_link": False,
                    'stayed': False,
        }
        if action == 4:
            blocked['stayed'] = True
            return blocked, None, prev

        # bounds check
        if not self.in_bounds(*intended):
            # print("Blocked by out of bound")
            blocked['oob'] = True
            return blocked, None, prev

        # check explicit node-level faults
        if intended in self.faulty_nodes:
            # print("Blocked by faulty")
            blocked['faulty'] = True
            return blocked, None, prev

        # neighbor input port that must be acquired: opp = (action + 2) % 4
        opp_port = (action + 2) % 4
        # If that specific (neighbor, opp_port) was flagged sustained-faulty, block
        if ((intended, opp_port) in self.sustained_faulty_ports):
            # print("sustained", self.sustained_faulty_ports)
            # print(f"Blocked by sustained faulty at node {intended, opp_port}")
            blocked['sustained'] = True
            return blocked, None, prev

        # neighbor input port that must be acquired: opp = (action + 2) % 4
        opp_port = (action + 2) % 4
        # availability at neighbor's input port
        avail = self.vc_free.get((intended, opp_port), 0) - self.vc_reserved.get((intended, opp_port), 0)
        if avail <= 0:
            # blocked due to no VC on neighbor in_port
            # print("Blocked due to no available vc")
            blocked['no_vc'] = True
            return blocked, None, prev

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
            # print("Blocked by busy link")
            blocked['busy_link'] = True
            return blocked, None, prev

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
        # print(f"Traversed Link: {traversed_link}, Occupied Links: {self.occupied_links}")
        return blocked, traversed_link, intended
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
    # compute congestion in neighbourhood; return neighbor congestion levels per direction
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
            # print(f"Node {node_pos}, Congestion History: {self.vc_congestion_history.get((neigh, opp_port), 0)}")

            st = self.vc_state.get((neigh, opp_port), {'no_vc_streak': 0, 'ema': 0.0, 'history': 0, 'sustained': False})
            if st['no_vc_streak'] >= self.VC_FAULT_PERSISTENCE:
                neighbor_congs[a] = 3
                continue
            if st['history'] >= self.SUSTAINED_CONGESTION_THRESHOLD:
                neighbor_congs[a] = 3
                continue
            if st['sustained'] or neigh in self.faulty_nodes:
                neighbor_congs[a] = 3
                continue

            ema = st['ema']
            if ema >= 0.66:
                neighbor_congs[a] = 2
            elif ema >= 0.33:
                neighbor_congs[a] = 1
            else:
                neighbor_congs[a] = 0

        return neighbor_congs
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
        non_existing_neigh = 5 - len(neigh_nodes)
        # print(f"Node: {center_pos}, Neigh Nodes: {neigh_nodes}")
        node_area = max(1, len(neigh_nodes))
        total_hist = sum(self.node_visit_count.get(n, 0) for n in neigh_nodes)
        # print(f"Node: {center_pos}, Total Hist: {total_hist}")
        avg_hist = total_hist / node_area
        hist_score = float(math.tanh(avg_hist / max(1.0, self.HISTORY_SCALE)))

        active_in_neigh = -1
        if active_packets is not None:
            for p in active_packets:
                if abs(p.pos[0] - cx) + abs(p.pos[1] - cy) <= radius:
                    active_in_neigh += 1

        active_scale = self.NUM_VCS * 2
        active_score = float(math.tanh(active_in_neigh/ active_scale))
        # print(f"Active in Neigh: {active_in_neigh} ")

        W_HIST, W_ACTIVE = 0.0, 1.0
        combined = W_HIST * hist_score + W_ACTIVE * active_score
        # print(f"Hist {hist_score}, Active: {active_score}, Combined: {combined}")
        fault_in_neigh = 0
        for node in neigh_nodes:
            if node in self.faulty_nodes:
                # print("Faulty", node)
                fault_in_neigh += 1

        for port in [0, 1, 2, 3]:
            neigh_node = center_pos[0] + self.DELTAS[port][0], center_pos[1] + self.DELTAS[port][1]
            if (center_pos, port) in self.sustained_faulty_ports and neigh_node not in self.faulty_nodes:
                fault_in_neigh += 1
                # print("Sustained Faulty", neigh_node, "Port", port)
        # fault_in_neigh += non_existing_neigh
        MAX_FAULT = 5

        fault_score = float(math.tanh(fault_in_neigh / MAX_FAULT))
        # print(F"Fault Score: {fault_score}, Fault in neigh: {fault_in_neigh}")

        return float(min(max(combined, 0.0), 1.0)), fault_score, {'hist': hist_score, 'active': active_score, 'fault': fault_in_neigh, 'active_in_neigh': active_in_neigh}
