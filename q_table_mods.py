import pickle
# import scipy.stats as ss
import numpy as np


def pickle_to_txt(pickle_file, text_file):
    # Load the pickle file
    with open(pickle_file, 'rb') as pf:
        data = pickle.load(pf)

    # Write the data to a text file
    with open(text_file, 'w') as tf:
        # Convert the data to a string and write it
        tf.write(str(data))

    print(f"Data from {pickle_file} has been saved to {text_file}")


def modified_q_table(q_table, index_file):
    new_q_table = {}
    max_q_table = {}
    for key, value in q_table.items():
        max_q_table[(key)] = int(np.argmax(value))
        # new_q_table[(key)] = [int(x) for x in ss.rankdata(value)]
        # with open(rank_file, 'w') as tf:
        #     # Convert the data to a string and write it
        #     tf.write(str(new_q_table))

        with open(index_file, 'w') as tf:
            # Convert the data to a string and write it
            tf.write(str(max_q_table))


# pickle_file = "D:\\QLearningNOCNew\\q_table-1766467357.pickle"
#
# text_file = "D:\\QLearningNOCNew\\q_table-1766467357.txt"
#
# pickle_to_txt(pickle_file, text_file)

import matplotlib.pyplot as plt

# Mesh size
N = 5

# Create node list
# Convention: x = row, y = column
nodes = []
for x in range(N):
    for y in range(N):
        node_id = x * N + y
        nodes.append((node_id, x, y))

# Create figure
plt.figure(figsize=(6, 6))
plt.title("5x5 Mesh (x = row, y = column)")
plt.gca().set_aspect('equal')

# Set axes
plt.xticks(range(N))
plt.yticks(range(N))
plt.xlim(-0.5, N - 0.5)
plt.ylim(-0.5, N - 0.5)
plt.xlabel("column (y)")
plt.ylabel("row (x)")
plt.grid(True, linestyle='--', linewidth=0.5)

# Draw mesh links (4-neighbour connectivity)
for _, x, y in nodes:
    # Right neighbour
    if y + 1 < N:
        plt.plot([y, y + 1], [x, x])
    # Down neighbour
    if x + 1 < N:
        plt.plot([y, y], [x, x + 1])

# Plot nodes
x_coords = [y for _, _, y in nodes]  # column on x-axis
y_coords = [x for _, x, _ in nodes]  # row on y-axis
plt.scatter(x_coords, y_coords, s=120)

# Annotate nodes with id and (x,y)
for node_id, x, y in nodes:
    plt.text(
        y + 0.05,
        x - 0.15,
        f"{node_id}\n({x},{y})",
        fontsize=9
    )

# Invert y-axis so row 0 is at the top (matrix view)
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
