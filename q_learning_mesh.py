import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import networkx as nx
from environment import Node, Link
# import matplotlib.pyplot as plt
# import seaborn as sns

style.use("ggplot")

SIZE = 5
HM_EPISODES = 10001
MOVE_PENALTY = 10
OBSTACLE = 200
DEST_REWARD = 50
EPSILON = 0.95
EPS_DECAY = 0.999
SHOW_EVERY = 1000
EXPLORATION_STEPS = 5000
SHOW_LAST = 100

# start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

SOURCE_N = 1
DEST_N = 2
# ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0)}

# --- Reward / fault / congestion hyperparameters ---
W_MOVE = 1.0                # weight for base move penalty
W_DIRECTION = 20.0          # weight for reward / penalty for moving closer / farther
W_FAULT = 60.0              # penalty weight for entering or attempting a faulty node/link
W_CONGESTION = 5.0          # extra penalty proportional to node/link usage (congestion)
W_DISTANCE_PROGRESS = 10.0  # scaling factor for progress proportional to distance closed

# fault penalty constant (higher than regular move penalty)
FAULT_MOVE_PENALTY = 50

# optional: whether attempting a blocked move counts toward congestion
BLOCKED_INCREASES_CONGESTION = True

# initialize congestion counters for nodes and links
from collections import defaultdict
node_visit_count = defaultdict(int)    # counts times agent visited a node (successful visit)
link_visit_count = defaultdict(int)    # counts times a link was traversed (successful traversal)

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def compute_reward(prev_node, new_node, dest,
                   traversed_link,
                   was_blocked,
                   faulty_nodes, faulty_links,
                   node_visit_count, link_visit_count):
    """
    Returns: (reward, details_dict)
    details_dict contains component breakdown for debugging/analysis.
    """
    details = {}

    # 1) Base move penalty
    if new_node == prev_node:
        base_move = MOVE_PENALTY + 10  # stayed in place extra cost (your existing logic)
    else:
        base_move = MOVE_PENALTY
    details['base_move'] = -W_MOVE * base_move

    # 2) Direction/progress reward: measure change in Manhattan distance to destination
    prev_dist = manhattan(prev_node, dest)
    new_dist = manhattan(new_node, dest)
    dist_progress = prev_dist - new_dist  # positive if moved closer
    # normalize progress by max possible distance (grid diameter = 2*(SIZE-1))
    max_dist = 2 * (SIZE - 1) if SIZE > 1 else 1
    norm_progress = dist_progress / max_dist
    direction_component = W_DIRECTION * norm_progress
    # scale with absolute progress too (optionally)
    details['direction'] = direction_component

    # 3) Fault penalty (if move was blocked OR new_node is faulty OR traversed_link is faulty)
    fault_component = 0.0
    if was_blocked:
        fault_component = -W_FAULT
    else:
        if new_node in faulty_nodes:
            fault_component = -W_FAULT
        elif traversed_link in faulty_links:
            fault_component = -W_FAULT
    details['fault'] = fault_component

    # 4) Congestion penalty: based on prior visit counts (higher visited nodes/links cost more)
    node_count = node_visit_count.get(new_node, 0)
    link_count = link_visit_count.get(traversed_link, 0) if traversed_link is not None else 0
    congestion_component = -W_CONGESTION * (node_count + link_count)
    details['congestion'] = congestion_component

    # 5) Destination reward (override / big positive)
    if new_node == dest:
        dest_comp = DEST_REWARD
    else:
        dest_comp = 0
    details['dest'] = dest_comp

    # total reward
    total_reward = details['base_move'] + details['direction'] + details['fault'] + details['congestion'] + details['dest']
    details['total'] = total_reward
    details['prev_dist'] = prev_dist
    details['new_dist'] = new_dist
    details['norm_progress'] = norm_progress

    return total_reward, details


# add near top of file with other imports
import random

# constants (add near your other constants)
MAX_NODE_FAULTS = 4
MAX_LINK_FAULTS = 4
FAULT_MOVE_PENALTY = 50  # extra penalty when trying to move into a faulty router / across faulty link

# helper: ensure q_table has an entry for an observation key
def ensure_q_key(q_table, key):
    if key not in q_table:
        q_table[key] = [np.random.uniform(-5, 0) for _ in range(4)]

# helper: return list of all valid grid coordinates (same range used in init_q_table)
def all_grid_nodes():
    return [(x, y) for x in range(-SIZE + 1, SIZE) for y in range(-SIZE + 1, SIZE)]

# helper: generate all adjacent-links (undirected) in the grid
def all_adjacent_links():
    nodes = all_grid_nodes()
    links = set()
    for (x, y) in nodes:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxn = (x + dx, y + dy)
            if nxn in nodes:
                links.add(frozenset({(x, y), nxn}))
    return list(links)

ALL_POSSIBLE_LINKS = all_adjacent_links()

# main: generate faults for an episode (avoid source and dest)
def generate_random_faults(source, dest,
                           max_node_faults=MAX_NODE_FAULTS,
                           max_link_faults=MAX_LINK_FAULTS):
    nodes_space = [n for n in all_grid_nodes() if n != source and n != dest]

    node_fault_count = np.random.randint(0, max_node_faults + 1)
    faulty_nodes = set(random.sample(nodes_space, k=node_fault_count)) if node_fault_count > 0 else set()

    # possible links: exclude any link that includes source or dest (optional)
    possible_links = [L for L in ALL_POSSIBLE_LINKS if (source not in L and dest not in L)]
    link_fault_count = np.random.randint(0, max_link_faults + 1)
    faulty_links = set(random.sample(possible_links, k=link_fault_count)) if link_fault_count > 0 else set()

    return faulty_nodes, faulty_links


def init_q_table(start_q_table=None):
    if start_q_table is None:
        q_table = {}
        # start = 0
        # total_nodes = SIZE ** 2
        for x in range(-SIZE + 1, SIZE):
            for y in range(-SIZE + 1, SIZE):
                q_table[(x, y)] = [np.random.uniform(-5, 0) for i in range(4)]

    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    return q_table


episode_rewards = []
total_moves = []
expected_moves = []

# links = init_links()
q_table = init_q_table()
print("Q table", q_table)

def get_node_index(x, y):
    return x * SIZE + y


for episode in range(HM_EPISODES):

    start_node = Node()
    end_node = Node()
    source = (start_node.x, start_node.y)
    dest = (end_node.x, end_node.y)
    expected_moves.append(np.abs(start_node.x - end_node.x) + np.abs(start_node.y - end_node.y))
    # faulty nodes and links
    faulty_nodes, faulty_links = generate_random_faults(source, dest)
    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {EPSILON}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0

    moves = 200
    moves_path = []
    moves_path.append(source)
    cost = []
    if episode >= HM_EPISODES - SHOW_LAST:
        print("Source:", source, "Destination:", dest)
    # for i in range(200):
    #     current_node = (start_node.x, start_node.y)
    #     obs = dest[0] - current_node[0], dest[1] - current_node[1]
    #     if np.random.random()>EPSILON and episode > EXPLORATION_STEPS:
    #         action = np.argmax(q_table[obs])
    #     else:
    #         action = np.random.randint(0, 4)
    #     start_node.action(action)
    #     new_node = (start_node.x, start_node.y)
    #     if current_node == new_node:
    #         weight = MOVE_PENALTY + 10
    #     else:
    #         weight = MOVE_PENALTY
    #     cost.append(weight)
    #
    #     if new_node == dest:
    #         reward = DEST_REWARD
    #         moves = i
    #     else:
    #         reward = -weight
    #     new_obs = dest[0] - current_node[0], dest[1] - current_node[1]
    #     max_future_q = np.max(q_table[new_obs])
    #     current_q = q_table[obs][action]
    #
    #     # q_table Update
    #     if new_node == dest:
    #         new_q = DEST_REWARD
    #     else:
    #         new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    #
    #     q_table[obs][action] = new_q
    #
    #     if episode >= HM_EPISODES - SHOW_LAST:
    #         moves_path.append(new_node)
    #
    #     # show the env
    #     if show:
    #         env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    #         env[end_node.y][end_node.x] = d[DEST_N]
    #         env[start_node.y][start_node.x] = d[SOURCE_N]
    #         # env[enemy.y][enemy.x] = d[ENEMY_N]
    #
    #         img = Image.fromarray(env, "RGB")
    #         img = img.resize((300, 300), resample=Image.BOX)
    #         cv2.imshow("", np.array(img))
    #         if reward == DEST_REWARD or reward == - OBSTACLE:
    #             if cv2.waitKey(500) & 0xFF == ord("q"):
    #                 break
    #         else:
    #             if cv2.waitKey(1) & 0xFF == ord("q"):
    #                 break
    #
    #     episode_reward += reward
    #     if new_node == dest:
    #         break

    for i in range(200):
        # --- before applying action: record current node ---
        current_node = (start_node.x, start_node.y)
        obs = dest[0] - current_node[0], dest[1] - current_node[1]

        # choose action
        if np.random.random() > EPSILON and episode > EXPLORATION_STEPS:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        # apply action
        prev_node = (start_node.x, start_node.y)
        start_node.action(action)
        new_node = (start_node.x, start_node.y)
        traversed_link = frozenset({prev_node, new_node}) if prev_node != new_node else None

        # determine if move was blocked by fault (same as earlier approach)
        was_blocked = False
        if new_node in faulty_nodes:
            was_blocked = True
        if traversed_link is not None and traversed_link in faulty_links:
            was_blocked = True

        # If blocked, revert position (keeping Node class unchanged)
        if was_blocked:
            # revert
            start_node.x, start_node.y = prev_node
            new_node = prev_node
            traversed_link = None  # no successful traversal

        # compute the dynamic reward using the new function
        reward, reward_details = compute_reward(prev_node=prev_node,
                                                new_node=new_node,
                                                dest=dest,
                                                traversed_link=traversed_link,
                                                was_blocked=was_blocked,
                                                faulty_nodes=faulty_nodes,
                                                faulty_links=faulty_links,
                                                node_visit_count=node_visit_count,
                                                link_visit_count=link_visit_count)

        # update congestion counters for successful traversals / visits
        if not was_blocked:
            node_visit_count[new_node] += 1
            if traversed_link is not None:
                link_visit_count[traversed_link] += 1
        else:
            if BLOCKED_INCREASES_CONGESTION:
                # if you want blocked attempts to also increase congestion counters:
                node_visit_count[prev_node] += 1
                # do not increment link count because traversal didn't succeed

        # For Q-learning: new_obs must reflect the post-action state (new_node)
        new_obs = dest[0] - new_node[0], dest[1] - new_node[1]
        ensure_q_key(q_table, obs)
        ensure_q_key(q_table, new_obs)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if new_node == dest:
            new_q = DEST_REWARD
            moves = i
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        if episode >= HM_EPISODES - SHOW_LAST:
            moves_path.append(new_node)

        # show the env (optionally mark faulty nodes/links)
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[end_node.y][end_node.x] = d[DEST_N]
            env[start_node.y][start_node.x] = d[SOURCE_N]

            # visualize faulty nodes (yellow-ish) and blocked links (red) — adjust as needed
            for fn in faulty_nodes:
                fx, fy = fn
                env[fy][fx] = (0, 0, 255)  # blue (you can pick any)
            # Note: link visualization to env array is approximate (not precise line drawing).
            # If you want clear visuals, create a Matplotlib overlay or draw lines on the image.

            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300), resample=Image.BOX)
            cv2.imshow("", np.array(img))
            if reward == DEST_REWARD or reward == - OBSTACLE:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if new_node == dest:
            break

    if episode >= HM_EPISODES - SHOW_LAST:
        print("Path:", moves_path)

    avg_cost = sum(cost)/(moves + 1)
    total_moves.append(moves + 1)

    if show:
        print("AVG Total Moves", np.mean(total_moves[-SHOW_EVERY:]))
        print("AVG Expected Moves", np.mean(expected_moves[-SHOW_EVERY:]))
        print("AVG Cost", avg_cost)
    episode_rewards.append(episode_reward)
    if episode > EXPLORATION_STEPS:
        EPSILON *= EPS_DECAY


moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

moves_avg = np.convolve(total_moves, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
expected_avg_moves = np.convolve(expected_moves, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Cummulative Reward")
plt.xlabel("Episodes")
plt.show()

plt.plot([i for i in range(len(moves_avg))], moves_avg)
plt.plot([i for i in range(len(expected_avg_moves))], expected_avg_moves)
plt.ylabel(f"Total Steps")
plt.xlabel("Episodes")
plt.show()

with open(f"q_table-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

# train()

def visualize_qtable():
    pass
