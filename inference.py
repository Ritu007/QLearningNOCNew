
import numpy as np
from environment import *

q_table = {(-4, -4): 0, (-4, -3): 0, (-4, -2): 3, (-4, -1): 0, (-4, 0): 3, (-4, 1): 2, (-4, 2): 3, (-4, 3): 3, (-4, 4): 2, (-3, -4): 3, (-3, -3): 3, (-3, -2): 0, (-3, -1): 0, (-3, 0): 3, (-3, 1): 2, (-3, 2): 2, (-3, 3): 2, (-3, 4): 2, (-2, -4): 0, (-2, -3): 0, (-2, -2): 0, (-2, -1): 0, (-2, 0): 3, (-2, 1): 2, (-2, 2): 2, (-2, 3): 2, (-2, 4): 2, (-1, -4): 0, (-1, -3): 0, (-1, -2): 0, (-1, -1): 0, (-1, 0): 3, (-1, 1): 2, (-1, 2): 2, (-1, 3): 2, (-1, 4): 3, (0, -4): 0, (0, -3): 0, (0, -2): 0, (0, -1): 0, (0, 0): 1, (0, 1): 2, (0, 2): 2, (0, 3): 2, (0, 4): 2, (1, -4): 0, (1, -3): 0, (1, -2): 0, (1, -1): 0, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 2, (2, -4): 0, (2, -3): 0, (2, -2): 0, (2, -1): 0, (2, 0): 1, (2, 1): 1, (2, 2): 1, (2, 3): 1, (2, 4): 2, (3, -4): 0, (3, -3): 1, (3, -2): 0, (3, -1): 0, (3, 0): 1, (3, 1): 1, (3, 2): 1, (3, 3): 2, (3, 4): 1, (4, -4): 0, (4, -3): 0, (4, -2): 1, (4, -1): 1, (4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 2}


route_table = {}
for i in range(5):
    for j in range(5):
        for k in range(5):
            for l in range(5):

                source = Node(i, j, inference=True)
                destination = Node(k, l, inference=True)
                current = source
                source_node = (source.x, source.y)
                current_node = (source.x, source.y)
                dest_node = (destination.x, destination.y)
                route_table[(source_node, dest_node)] = []
                while current_node != dest_node:
                    choice = q_table[(dest_node[0] - current_node[0]), (dest_node[1] - current_node[1])]
                    # print(current_node)

                    print(choice)
                    current.action(choice)
                    current_node = current.x, current.y
                    route_table[(source_node, dest_node)].append(current_node)
                    print(current_node)

route_file ="route_5x5.txt"
with open(route_file, 'w') as rf:
# Convert the data to a string and write it
    rf.write(str(route_table))


def one_path_infer():
    source = Node(0, 0, inference=True)
    destination = Node(4, 4, inference=True)
    current = source
    current_node = (source.x, source.y)
    dest_node = (destination.x, destination.y)

    # current_node = source

    # print(source)
    while current_node != dest_node:
        # print((dest_node[0] - current_node[0], dest_node[1] - current_node[1]))
        choice = q_table[(dest_node[0] - current_node[0], dest_node[1] - current_node[1])]
        # print(current_node)

        print(choice)
        current.action(choice)
        current_node = current.x, current.y

        print(current_node)