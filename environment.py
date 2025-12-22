import numpy as np


# class Network:
#     def __init__(self, size=5):
#         self.size = size
#         self.routers = []
#         self.edges = []
#
#     def init_routers(self):
#         for i in range(self.size):
#             for j in range()

class Node:
    def __init__(self, x=0, y=0, inference=False, size=5):
        self.size = size
        self.edges = [-1, -1, -1, -1]
        if inference:
            self.x = x
            self.y = y

        else:
            self.x = np.random.randint(0, self.size)
            self.y = np.random.randint(0, self.size)
        self.move_weight = 0

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        if choice == 0:
            self.move(del_x=0, del_y=-1)
            self.move_weight = np.random.randint(0, 5)
        elif choice == 1:
            self.move(del_x=1, del_y=0)
            self.move_weight = np.random.randint(0, 5)
        elif choice == 2:
            self.move(del_x=0, del_y=1)
            self.move_weight = np.random.randint(0, 5)
        elif choice == 3:
            self.move(del_x=-1, del_y=0)
            self.move_weight = np.random.randint(0, 5)

    def move(self, del_x=False, del_y=False):
        self.x += del_x
        self.y += del_y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1

        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1


class Link:
    def __init__(self, id, node_a, node_b, weight):
        self.id = id
        self.node_a = node_a
        self.node_b = node_b
        self.weight = weight

    def __repr__(self):
        return f"{self.node_a} --- {self.node_b}: {self.weight}"