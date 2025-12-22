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

