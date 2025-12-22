import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import networkx as nx
# from environment import Node, Link
from new_environment import Network, Data
# import matplotlib.pyplot as plt
# import seaborn as sns
import random
from q_table_mods import *

style.use("ggplot")

SIZE = 4
HM_EPISODES = 100000
MOVE_PENALTY = 10
OBSTACLE = 200
DEST_REWARD = 50
EPSILON = 0.95
EPS_DECAY = 0.9999
SHOW_EVERY = 1000
EXPLORATION_STEPS = 25000
SHOW_LAST = 1000
MAX_STEPS = 200

# start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

SOURCE_N = 1
DEST_N = 2
# ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0)}

faulty_routers =  [5, 10]
faulty_links = [6, 15]

def init_q_table(start_q_table=None):
    if start_q_table is None:
        q_table = {}
        # start = 0
        # total_nodes = SIZE ** 2
        for x in range(SIZE * SIZE):
            for y in range(SIZE * SIZE):
                q_table[(x, y)] = [float(np.random.uniform(-5, 0)) for i in range(4)]

    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    return q_table


network = Network(faulty_routers, faulty_links, size=SIZE)

episode_rewards = []
total_moves = []
expected_moves = []

q_table = init_q_table()
print("Q table", q_table)

for episode in range(HM_EPISODES):
    obs = network.reset()
    # das
    # obs = network.get_observation()
    # print(obs[0])
    episode_reward = 0
    episode_moves = 0

    for i in range(MAX_STEPS):
        if random.random() > EPSILON and episode > EXPLORATION_STEPS:
            action = int(np.argmax(q_table[obs[0]]))
        else:
            action = random.randint(0, 3)

        plt.ion()
        # if episode > (HM_EPISODES - SHOW_LAST):
        #     network.render_topology()
        # print("Action", action)
        new_obs, reward, done = network.step(action)
        episode_reward += reward

        max_future_q = float(np.max(q_table[new_obs[0]]))
        current_q = q_table[obs[0]][action]
        # q_table Update
        episode_moves = i
        if done:
            new_q = reward
            q_table[obs[0]][action] = new_q



            break
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[obs[0]][action] = new_q
            obs = new_obs
    plt.ioff()
    plt.show()

    # print("Episode Reward", episode_reward)
    episode_rewards.append(episode_reward)
    total_moves.append(episode_moves)

    # print("AVG Total Moves", np.mean(total_moves[-SHOW_EVERY:]))
    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}, Reward: {episode_reward}, Total Moves: {episode_moves}, Average Steps: {np.mean(total_moves[-SHOW_EVERY:])}")

    if episode > EXPLORATION_STEPS:
        EPSILON *= EPS_DECAY

# moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
#
# moves_avg = np.convolve(total_moves, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
# # expected_avg_moves = np.convolve(expected_moves, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
#
# plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.ylabel(f"reward {SHOW_EVERY} ma")
# plt.xlabel("episode #")
# plt.show()
#
# plt.plot([i for i in range(len(moves_avg))], moves_avg)
# # plt.plot([i for i in range(len(expected_avg_moves))], expected_avg_moves)
# plt.ylabel(f"moves {SHOW_EVERY}")
# plt.xlabel("episode #")
# plt.show()

q_table_file = f"Q_Tables\\Faulty Routers and Links\\q_table-{SIZE}x{SIZE}_F_router_{faulty_routers}_F_link_{faulty_links}"

with open(f"{q_table_file}.pickle", "wb") as f:
    pickle.dump(q_table, f)

pickle_to_txt(f"{q_table_file}.pickle", f"{q_table_file}.txt",)
modified_q_table(q_table, f"{q_table_file}-index.txt")