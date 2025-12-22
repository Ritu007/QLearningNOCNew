import time
import math
import numpy as np
from environment import *
from PIL import Image
import cv2
# import matplotlib.pyplot as plt
# import pickle
# from matplotlib import style

q_table = {(-4, -4): 0, (-4, -3): 0, (-4, -2): 3, (-4, -1): 0, (-4, 0): 3, (-4, 1): 2, (-4, 2): 3, (-4, 3): 3, (-4, 4): 2, (-3, -4): 3, (-3, -3): 3, (-3, -2): 0, (-3, -1): 0, (-3, 0): 3, (-3, 1): 2, (-3, 2): 2, (-3, 3): 2, (-3, 4): 2, (-2, -4): 0, (-2, -3): 0, (-2, -2): 0, (-2, -1): 0, (-2, 0): 3, (-2, 1): 2, (-2, 2): 2, (-2, 3): 2, (-2, 4): 2, (-1, -4): 0, (-1, -3): 0, (-1, -2): 0, (-1, -1): 0, (-1, 0): 3, (-1, 1): 2, (-1, 2): 2, (-1, 3): 2, (-1, 4): 3, (0, -4): 0, (0, -3): 0, (0, -2): 0, (0, -1): 0, (0, 0): 1, (0, 1): 2, (0, 2): 2, (0, 3): 2, (0, 4): 2, (1, -4): 0, (1, -3): 0, (1, -2): 0, (1, -1): 0, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 2, (2, -4): 0, (2, -3): 0, (2, -2): 0, (2, -1): 0, (2, 0): 1, (2, 1): 1, (2, 2): 1, (2, 3): 1, (2, 4): 2, (3, -4): 0, (3, -3): 1, (3, -2): 0, (3, -1): 0, (3, 0): 1, (3, 1): 1, (3, 2): 1, (3, 3): 2, (3, 4): 1, (4, -4): 0, (4, -3): 0, (4, -2): 1, (4, -1): 1, (4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 2}
q_table_rank = {(-4, -4): [4, 1, 2, 3], (-4, -3): [4, 1, 2, 3], (-4, -2): [3, 1, 2, 4], (-4, -1): [4, 1, 2, 3], (-4, 0): [2, 1, 3, 4], (-4, 1): [2, 1, 4, 3], (-4, 2): [2, 1, 3, 4], (-4, 3): [2, 1, 3, 4], (-4, 4): [2, 1, 4, 3], (-3, -4): [3, 2, 1, 4], (-3, -3): [3, 1, 2, 4], (-3, -2): [3, 1, 2, 3], (-3, -1): [3, 2, 1, 3], (-3, 0): [3, 1, 2, 4], (-3, 1): [2, 1, 3, 3], (-3, 2): [2, 1, 3, 3], (-3, 3): [1, 2, 4, 3], (-3, 4): [1, 2, 4, 3], (-2, -4): [4, 2, 1, 3], (-2, -3): [3, 2, 1, 3], (-2, -2): [3, 1, 2, 3], (-2, -1): [3, 1, 2, 3], (-2, 0): [2, 3, 1, 4], (-2, 1): [2, 1, 3, 3], (-2, 2): [2, 1, 3, 3], (-2, 3): [1, 2, 3, 3], (-2, 4): [1, 2, 4, 3], (-1, -4): [4, 2, 1, 3], (-1, -3): [3, 2, 1, 3], (-1, -2): [3, 1, 2, 3], (-1, -1): [3, 1, 2, 3], (-1, 0): [1, 3, 2, 4], (-1, 1): [1, 2, 3, 3], (-1, 2): [2, 1, 3, 3], (-1, 3): [1, 2, 3, 3], (-1, 4): [1, 2, 3, 4], (0, -4): [4, 2, 1, 3], (0, -3): [4, 3, 2, 1], (0, -2): [4, 3, 1, 2], (0, -1): [4, 1, 3, 2], (0, 0): [1, 4, 2, 3], (0, 1): [2, 1, 4, 3], (0, 2): [2, 1, 4, 3], (0, 3): [2, 1, 4, 3], (0, 4): [1, 3, 4, 2], (1, -4): [4, 3, 1, 2], (1, -3): [3, 3, 1, 2], (1, -2): [3, 3, 2, 1], (1, -1): [3, 3, 2, 1], (1, 0): [1, 4, 3, 2], (1, 1): [1, 3, 3, 2], (1, 2): [1, 3, 3, 2], (1, 3): [1, 3, 3, 2], (1, 4): [1, 3, 4, 2], (2, -4): [4, 3, 1, 2], (2, -3): [3, 3, 1, 2], (2, -2): [3, 3, 1, 2], (2, -1): [3, 3, 2, 1], (2, 0): [2, 4, 3, 1], (2, 1): [2, 3, 3, 1], (2, 2): [2, 3, 3, 1], (2, 3): [2, 3, 3, 1], (2, 4): [1, 3, 4, 2], (3, -4): [4, 3, 1, 2], (3, -3): [3, 4, 2, 1], (3, -2): [3, 3, 1, 2], (3, -1): [3, 3, 2, 1], (3, 0): [2, 4, 3, 1], (3, 1): [2, 3, 3, 1], (3, 2): [2, 3, 3, 1], (3, 3): [2, 3, 4, 1], (3, 4): [1, 4, 3, 2], (4, -4): [4, 3, 1, 2], (4, -3): [4, 3, 2, 1], (4, -2): [3, 4, 2, 1], (4, -1): [3, 4, 2, 1], (4, 0): [2, 4, 3, 1], (4, 1): [2, 4, 3, 1], (4, 2): [2, 4, 3, 1], (4, 3): [2, 4, 3, 1], (4, 4): [1, 3, 4, 2]}

faulty_routers = [(1, 3), (1, 2), (0, 1)]
route_table = {}

start_N = 1
ending_N = 2
# ENEMY_N = 3
# Colors (BGR format for OpenCV)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # Source node
RED = (0, 0, 255)  # Destination node
BLUE = (255, 0, 0)  # Regular links
ORANGE = (0, 165, 255)  # Diagonal links
GREY = (137, 137, 137) #Faulty
YELLOW = (225, 225, 0)

d = {1: (255, 175, 0),
     2: (0, 255, 0)}
show = True
NODE_SIZE = 50
SIZE = 5



def get_second_max(arr):
    first_val = second_val = float('-inf')
    first_index = second_index = -1

    # Iterate through the array once
    for i, val in enumerate(arr):
        if val > first_val:
            # Update second highest before updating the highest
            second_val, second_index = first_val, first_index
            first_val, first_index = val, i
        elif val > second_val:
            second_val, second_index = val, i

    return second_val, second_index

def isnotingrid(node):
    if node[0] > SIZE - 1 or node[0] < 0 or node[1] > SIZE - 1 or node[1] < 0:
        return True
    return False


def restricted_nodes(current_node, prev_node):
    left_node = (current_node[0], current_node[1] - 1)
    down_node = (current_node[0] + 1, current_node[1])
    right_node = (current_node[0], current_node[1] + 1)
    up_node = (current_node[0] - 1, current_node[1])

    restrict = []
    if left_node in faulty_routers or np.array_equal(left_node, prev_node) or isnotingrid(left_node):
        restrict.append(0)
        # print("0 rest")
    if down_node in faulty_routers or np.array_equal(down_node, prev_node) or isnotingrid(down_node):
        restrict.append(1)
        # print("1 rest")
    if right_node in faulty_routers or np.array_equal(right_node, prev_node) or isnotingrid(right_node):
        restrict.append(2)
        # print("2 rest")
    if up_node in faulty_routers or np.array_equal(up_node, prev_node) or isnotingrid(up_node):
        restrict.append(3)
        # print("3 rest")
    return restrict


def get_closest_node(source, destination, faulty_rows, faulty_columns):
    available_nodes = []
    tentative_nodes = []
    for i in range(len(faulty_rows)):
        for j in range(len(faulty_columns)):
            if faulty_columns[j] == 0:
                if faulty_rows[i] == 0:
                    if not np.array_equal((i, j), source):
                        available_nodes.append((i, j))
                else:
                    if (i, j) not in faulty_routers:
                        tentative_nodes.append((i, j))
            else:
                if faulty_rows[i] == 0:
                    if (i, j) not in faulty_routers:
                        tentative_nodes.append((i, j))
                else:
                    break

    if len(available_nodes) == 0:
        available_nodes = tentative_nodes

    min_mh = 100
    best_node = destination
    for node in available_nodes:
        mh = 2 * mh_dist(node, source) + mh_dist(node, destination)
        if mh < min_mh:
            min_mh = mh
            best_node = node

    print("available nodes",available_nodes)
    print("tentative nodes", tentative_nodes)
    print("closest node", best_node)
    return best_node


def mh_dist(node_a, node_b):
    return abs(node_a[0] - node_b[0]) + abs(node_a[1] - node_b[1])


def check_path_congestion(destination_node, source_node):
    faulty_rows = np.zeros(SIZE)
    faulty_columns = np.zeros(SIZE)

    for router in faulty_routers:
        faulty_rows[router[0]] += 1
        faulty_columns[router[1]] += 1
    print("mh", mh_dist(destination_node, source_node))
    return faulty_rows, faulty_columns





def visualize_path(current_node, source_node, dest_node):
    if show:
        time.sleep(4)
        env = np.ones((250, 250, 3), dtype=np.uint8) * 255

        # Draw the grid and links
        for i in range(SIZE):
            for j in range(SIZE):
                # Compute node center
                x = j * NODE_SIZE + NODE_SIZE // 2
                y = i * NODE_SIZE + NODE_SIZE // 2

                # Draw vertical links (except last row)
                if i < SIZE - 1:
                    cv2.line(env, (x, y), (x, y + NODE_SIZE), ORANGE, 2)

                # Draw horizontal links (except last column)
                if j < SIZE - 1:
                    cv2.line(env, (x, y), (x + NODE_SIZE, y), ORANGE, 2)

                # Draw the node (circle)
                node_color = BLUE
                if (i, j) == source_node:
                    node_color = YELLOW
                if (i, j) == current_node:
                    node_color = GREEN  # Source node
                elif (i, j) == dest_node:
                    node_color = RED  # Destination node
                elif (i, j) in faulty_routers:
                    node_color = GREY

                cv2.circle(env, (x, y), 10, node_color, -1)

        # Resize for better display
        img = Image.fromarray(env, "RGB")
        img = img.resize((300, 300), resample=Image.BOX)
        cv2.imshow("Z-Mesh NoC", np.array(img))
        flag = False
        if current_node == dest_node:
            if cv2.waitKey(500) & 0xFF == ord("q"):
                flag = True
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                flag = True

        return flag


def one_path_infer():
    source = Node(0, 2, inference=True)
    destination = Node(2, 2, inference=True)
    current = source
    current_node = (source.x, source.y)
    dest_node = (destination.x, destination.y)
    source_node = (source.x, source.y)
    faulty_rows, faulty_columns = check_path_congestion(dest_node, source_node)
    print("faulty rows", faulty_rows, "faulty columns", faulty_columns)
    next_node = dest_node
    next_node = get_closest_node(source_node, dest_node, faulty_rows, faulty_columns)
    route_table[(source_node, dest_node)] = []
    # current_node = source
    prev_node = (-1, -1)
    print(f'source {source_node}, destination, {dest_node}')
    while current_node != dest_node:
        choice = q_table[(dest_node[0] - current_node[0]), (dest_node[1] - current_node[1])]
        # print(current_node)
        choice_rank = q_table_rank[(dest_node[0] - current_node[0]), (dest_node[1] - current_node[1])]
        print("state", (dest_node[0] - current_node[0]), (dest_node[1] - current_node[1]))
        print("q values", choice_rank)
        max_choice = np.argmax(np.array(choice_rank))

        print("max", max_choice)
        restrict = restricted_nodes(current_node, prev_node)
        print("restrict", restrict)
        max_val = -999
        max_index = -1
        for index in range(4):
            if index in restrict:
                continue
            else:
                if choice_rank[index] > max_val:
                    max_val = choice_rank[index]
                    max_index = index

        print("max", max_index)

        prev_node = current_node
        flag = visualize_path(current_node, source_node, dest_node)
        if flag:
            break
        current.action(max_index)
        current_node = current.x, current.y
        route_table[(source_node, dest_node)].append(current_node)
        print(current_node)

    flag = visualize_path(current_node, source_node, dest_node)


one_path_infer()