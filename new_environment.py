import numpy as np
import math
import matplotlib.pyplot as plt
import random

class Network:
    def __init__(self, faulty_routers, faulty_links,  size=4):
        self.size = size
        self.routers = []
        self.edges = []
        self.n_routers = size * size
        self.n_edges = 0
        self.faulty_routers = faulty_routers
        self.faulty_links = faulty_links
        self.data = []
        self.n_data = 1

    def init_routers(self):
        self.routers = []
        y_coordinate = 0
        x_coordinate = 0
        x_max = self.size - 1
        y_max = self.size - 1

        for i in range(self.n_routers):
            self.routers.append(Router(x_coordinate, y_max))
            if (x_coordinate == x_max):
                x_coordinate = 0
                y_max -= 1
            else:
                x_coordinate += 1

    def init_edges(self):
        # Define action order: 0=left,1=down,2=right,3=up
        self.edges = []
        self.n_edges = 0
        # for r in self.routers:
        #     # Initialize neighbor edges to -1 for each action
        #     r.edge = [-1, -1, -1, -1]
        #     r.neighbor = [-1, -1, -1, -1]

        for i in range(self.n_routers):
            # Compute squared distances to all other routers
            dis = []
            for j in range(self.n_routers):
                dx = self.routers[j].x - self.routers[i].x
                dy = self.routers[j].y - self.routers[i].y
                dis.append([(dx * dx + dy * dy), j, dx, dy])
            # Sort by distance ascending
            dis.sort(key=lambda x: x[0])

            # For each candidate neighbor at exactly distance 0.0625 (0.25 actual)
            for dist_sq, j, dx, dy in dis:
                # print("Dist sq", dist_sq)
                if i == j or dist_sq != 1:
                    continue
                # Determine direction index
                if dx < 0 and dy == 0:
                    direction = 0  # left
                elif dx == 0 and dy > 0:
                    direction = 1  # down
                elif dx > 0 and dy == 0:
                    direction = 2  # right
                elif dx == 0 and dy < 0:
                    direction = 3  # up
                else:
                    # not a direct cardinal neighbor
                    continue

                # print("dire", direction)
                # Create edge only once, and record its index
                if j not in self.routers[i].neighbor:
                    # add to neighbor lists
                    self.routers[i].neighbor[direction] = j
                    self.routers[j].neighbor[(direction + 2) % 4] = i

                    # instantiate Edge and record index
                    edge = Edge(min(i, j), max(i, j), math.sqrt(dist_sq))
                    self.edges.append(edge)
                    edge_idx = self.n_edges
                    self.n_edges += 1

                    # assign to neighbor_edges arrays
                    self.routers[i].edge[direction] = edge_idx
                    # compute opposite direction: (direction+2)%4
                    opp = (direction + 2) % 4
                    self.routers[j].edge[opp] = edge_idx

    def vis_routers(self):
        for i in range(self.n_routers):
            print(self.routers[i].x, self.routers[i].y)
            plt.scatter(self.routers[i].x, self.routers[i].y, color='red')
        plt.show()

    def vis_edges(self):
        for i in range(self.n_routers):
            plt.scatter(self.routers[i].x, self.routers[i].y, color='orange')
        for e in self.edges:
            plt.plot([self.routers[e.start].x, self.routers[e.end].x], [self.routers[e.start].y, self.routers[e.end].y], color='black')
        plt.show()

    def render_topology(self, pause_time=0.3):
        """
        routers: list of Router objects, each has .x, .y
        edges:   list of Edge objects, each has .start, .end
        data:    list of Data packets, each has .now, .edge
        """
        # print("data in render", self.data)
        plt.clf()  # clear previous frame
        # draw routers

        for router in self.routers:
            xs = router.x
            ys = router.y
            plt.scatter(xs, ys, color="lightgray", s=300, zorder=1)

            if router in self.faulty_routers:
                xf = xs
                yf = ys
                plt.scatter(xf, yf, color="black", s=250, zorder=2)
            plt.text(xs, ys, str(xs  + (self.size - 1 - ys) * self.size), color="red", ha="center", va="center", fontsize=8, zorder=5)


        # xs = [r.x for r in self.routers]
        # ys = [r.y for r in self.routers]
        # xf = [self.routers[r].x for r in self.faulty_routers]
        # yf = [self.routers[r].y for r in self.faulty_routers]
        # # print(f"xf {xf}, yf {yf}")
        # plt.scatter(xs, ys, color="lightgray", s=300, zorder=1)
        # plt.scatter(xf, yf, color="black", s=250, zorder=2)
        # plt.text(xs, ys, str(xs * self.size + ys), color="w", ha="center", va="center", fontsize=8, zorder=5)
        # draw links
        for idx, e in enumerate(self.edges):
            x0, y0 = self.routers[e.start].x, self.routers[e.start].y
            x1, y1 = self.routers[e.end].x, self.routers[e.end].y
            plt.plot([x0, x1], [y0, y1], color="k", zorder=0)
            if e.load > 0:
                plt.plot([x0, x1], [y0, y1], color="m", zorder=0)
            if e.is_faulty == 1:
                plt.plot([x0, x1], [y0, y1], color="y", zorder=0)

            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            plt.text(mx, my, str(idx), color="red", ha="center", va="center", fontsize=8, zorder=5)


        # draw packets
        for i, pkt in enumerate(self.data):
            # print(i)
            if pkt is None:
                continue
            if pkt.edge != -1:
                # packet is traversing an edge: draw at its midpoint
                e = self.edges[pkt.edge]
                x0, y0 = self.routers[e.start].x, self.routers[e.start].y
                x1, y1 = self.routers[e.end].x, self.routers[e.end].y
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            else:
                # packet is sitting at a router
                mx, my = self.routers[pkt.now].x, self.routers[pkt.now].y
                # Get target position
            tx, ty = self.routers[pkt.target].x, self.routers[pkt.target].y

            plt.scatter(mx, my, color="C1", s=100, zorder=4)
            # plt.text(mx, my, str(i), color="w", ha="center", va="center", fontsize=8, zorder=5)

            plt.scatter(tx, ty, color="red", s=180, zorder=3)
            # plt.text(tx, ty, str(i), color="w", ha="center", va="center", fontsize=8, zorder=5)

        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.pause(pause_time)


    def get_network_edges(self):
        print("total no of edges: ",self.n_edges)
        for i in range(self.n_routers):
            print("Router {}: x = {} y = {} neighbor = {} edges = {} two_hops = {} faulty = {}".format(i, self.routers[i].x, self.routers[i].y,
                                                                             self.routers[i].neighbor, self.routers[i].edge, self.routers[i].two_hops, self.routers[i].is_faulty))
        for i, e in enumerate(self.edges):
            print("Edge {}: start_point = {} end_point = {} length = {} load = {}".format(i, self.edges[i].start, self.edges[i].end,
                                                                                self.edges[i].len, self.edges[i].load))

    def create_data(self, source, destination, priority):
        data = Data(source, destination, np.random.random(), priority)
        # data.start_step = self.time_steps
        return data

    def set_data(self):
        packet_nos = self.n_data

        for i in range(packet_nos):
            destination = np.random.randint(self.n_routers)
            while destination in self.faulty_routers:
                destination = np.random.randint(self.n_routers)

            source = np.random.randint(self.n_routers)
            while source in self.faulty_routers or source == destination:
                source = np.random.randint(self.n_routers)

            self.data.append(self.create_data(source, destination, i))
            # self.available.pop(i)
        # Priority has been passed according to order of creation

        for j in range(packet_nos, self.n_data):
            self.data.append(None)

    def get_data_packet_info(self):

        for i in range(self.n_data):
            if self.data[i] is None:
                print("None")
                continue
            print(
                "Data Packet {}: start_point = {}, target  ={}, size = {}, priority = {}, edge = {}, steps = {}".format(
                    i, self.data[i].now,
                    self.data[i].target,
                    self.data[i].size,
                    self.data[i].priority,
                    self.data[i].edge,
                    self.data[i].total_steps))

    def set_faults(self):
        for i in range(len(self.routers)):
            if i in self.faulty_routers:
                self.routers[i].is_faulty = 1

        for j in range(len(self.edges)):
            if j in self.faulty_links:
                self.edges[j].is_faulty = 1



    def reset(self):
        self.init_routers()
        self.init_edges()
        self.set_faults()

        self.data = []
        # print(self.routers)
        # print(self.edges)
        # self.vis_routers()
        # self.vis_edges()
        # self.get_network_edges()
        # plt.ion()
        # self.render_topology(pause_time=2)
        # plt.ioff()
        # plt.show()
        self.set_data()
        # self.get_data_packet_info()
        obs = self.get_observation()

        return obs


    def get_observation(self):
        obs = []
        for data in self.data:
            ob = (data.now, data.target)
            obs.append(ob)
        return obs

    def step(self, action):
        rewards = 0
        done = False
        for packet in self.data:
            current_node = packet.now
            # print("Current Node", current_node)
            end_node = packet.target
            if current_node == end_node:
                rewards += 100
                done = True
                break
            next_node = self.routers[current_node].neighbor[action]
            # print("next Node", next_node)
            next_edge = self.routers[current_node].edge[action]
            # print("next edge", next_edge)

            if next_node < 0:
                rewards -= 20
            elif next_node in self.faulty_routers:
                rewards -= 50
            elif next_edge in self.faulty_links:
                rewards -= 50
            else:
                packet.now = next_node
                rewards -= 15
            # print("rewards", rewards)
            # print("done", done)
        observation = self.get_observation()
        return observation, rewards, done


class Router(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.neighbor = [-1,-1,-1,-1]
        self.edge = [-1,-1,-1,-1]
        self.is_faulty = 0
        self.two_hops = [-1, -1, -1,-1,-1,-1,-1,-1]
        # self.packets = []


class Edge(object):
    def __init__(self,x,y,l):
        self.start = x
        self.end = y
        self.len = int(l+1)
        self.load = 0
        self.is_faulty = 0


class Data(object):
    def __init__(self,x,y,size,priority):
        self.prev = None
        self.now = x
        self.target = y
        self.size = size
        self.priority = priority
        self.time = 0
        self.edge = -1
        self.neigh = [priority,-1,-1,-1]
        self.prev_dist = 0
        self.delivered = False
        self.is_empty = False
        # self.start_step = -1
        # self.end_step = -1
        self.total_steps = 0


# class Data:
#     def __init__(self, x=0, y=0, inference=False, size=5):
#         self.size = size
#
#         if inference:
#             self.x = x
#             self.y = y
#
#         else:
#             self.x = np.random.randint(0, self.size)
#             self.y = np.random.randint(0, self.size)
#         self.move_weight = 0
#
#     def __str__(self):
#         return f"{self.x}, {self.y}"
#
#     def __sub__(self, other):
#         return (self.x-other.x, self.y-other.y)
#
#     def action(self, choice):
#         if choice == 0:
#             self.move(del_x=0, del_y=-1)
#             self.move_weight = np.random.randint(0, 5)
#         elif choice == 1:
#             self.move(del_x=1, del_y=0)
#             self.move_weight = np.random.randint(0, 5)
#         elif choice == 2:
#             self.move(del_x=0, del_y=1)
#             self.move_weight = np.random.randint(0, 5)
#         elif choice == 3:
#             self.move(del_x=-1, del_y=0)
#             self.move_weight = np.random.randint(0, 5)
#
#     def move(self, del_x=False, del_y=False):
#         self.x += del_x
#         self.y += del_y
#
#         if self.x < 0:
#             self.x = 0
#         elif self.x > self.size - 1:
#             self.x = self.size - 1
#
#         if self.y < 0:
#             self.y = 0
#         elif self.y > self.size - 1:
#             self.y = self.size - 1
