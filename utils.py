# Plotting Delivery Rate, Average Hops, and Average Reward for multiple mesh sizes
# Using the data provided by the user in the conversation.
# - 5x5: full data for faults 0-5
# - 8x8: data for faults [3,6,10,13,24,36,48]; assume delivery=1.0 at 0 faults for delivery plot
# - 12x12: data for faults [7,12,14,21,28]; assume delivery=1.0 at 0 faults for delivery plot
# - 16x16: no detailed stats provided; assume delivery=1.0 at 0 faults (only a single point)
#
# Each figure will show curves for all available mesh sizes. For hops/reward, 16x16 is omitted due to missing data.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle, RegularPolygon
import pickle

# -------------------------
# drawing helpers
# -------------------------
NODE_BG = "#efefef"
NODE_FAULTY_BG = "#d9534f"   # red-ish
NODE_SUSTAIN_BG = "#f0ad4e"  # orange
NODE_BORDER = "#444444"
PACKET_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

PORT_MARKER_SIZE = 0.12  # fraction of cell



def plot_graphs():
    # 5x5 data (nodes = 25)
    mesh_5 = {
        "n": 5,
        "fault_pct": np.array([0, 5, 10, 15, 20]),
        "delivery": np.array(
            [1.0, 0.9981748429567544, 0.9874117931621954, 0.9808040975432871, 0.9449005655902207]),
        "hops": np.array(
            [2.357892551483109, 2.5547601498726555, 3.427826884638238, 4.515530886311313, 6.842420031360104]),
        "reward": np.array(
            [175.39340000000007, 133.48537000000013, 98.6394000000001, 31.095700000000058, -77.4797799999999])
    }
    # mesh_5["fault_pct"] = mesh_5["faults"] / (mesh_5["n"] ** 2) * 100

    # 8x8 data (nodes = 64)
    mesh_8 = {
        "n": 8,
        "fault_pct": np.array([0, 5, 10, 15, 20]),  # include 0 for delivery=1.0 assumption
        "delivery": np.array([1.0, 0.9879192788175537, 0.9738348743014373, 0.9297482937008217, 0.8926275127703373]),
        # For hops and reward we only have values for non-zero faults; we'll set a placeholder (nan) at index 0 so the line won't connect to it.
        "hops": np.array([4.759886127821959, 5.134085480222643, 5.991976952912512, 8.214863958191698, 9.89025388675675]),
        "reward": np.array([87.88445000000003, 68.30840000000009, 20.094989999999985, -64.54036999999994, -131.68726])
    }
    # mesh_8["fault_pct"] = mesh_8["faults"] / (mesh_8["n"] ** 2) * 100

    # 12x12 data (nodes = 144)
    mesh_12 = {
        "n": 12,
        # include 0 for delivery assumption; then faults where data is available
        "fault_pct": np.array([0, 5, 10, 15, 20]),
        "delivery": np.array([1.0, 0.9804745865727961, 0.9483745123537061, 0.9066231458910052, 0.8321101439381009]),
        "hops": np.array([6.25, 8.359709059121244, 9.642891158683922, 11.77801139654093, 14.55608026311955]),
        "reward": np.array([65.5, 22.083789999999986, -20.440809999999978, -75.07328999999989, -132.1332099999999])
    }
    # mesh_12["fault_pct"] = mesh_12["faults"] / (mesh_12["n"] ** 2) * 100

    # 16x16: only 0-fault assumed present (nodes = 256)
    mesh_16 = {
        "n": 16,
        "fault_pct": np.array([0, 5, 10, 15, 20]),
        "delivery": np.array([1.0, 0.9712294129706827, 0.9311691481571882, 0.8759109553582095, 0.7847818247385503]),
        "hops": np.array([8.5, 11.235688830074507, 12.8933749572133, 14.932755043717362, 17.829524597529335]),
        "reward": np.array([50.5, 9.916490000000028, -31.185530000000004, -74.03355000000009, -125.55476999999998])
    }
    # mesh_16["fault_pct"] = mesh_16["faults"] / (mesh_16["n"] ** 2) * 100

    # --- Plot 1: Fault % vs Delivery Rate ---
    plt.figure(figsize=(8, 5))
    plt.plot(mesh_5["fault_pct"], mesh_5["delivery"], marker='o', label='5x5')
    plt.plot(mesh_8["fault_pct"], mesh_8["delivery"], marker='o', label='8x8')
    plt.plot(mesh_12["fault_pct"], mesh_12["delivery"], marker='o', label='12x12')
    plt.plot(mesh_16["fault_pct"], mesh_16["delivery"], marker='o', label='16x16')
    plt.xlabel("Fault Percentage (%)")
    plt.ylabel("Delivery Rate")
    plt.title("Fault % vs Delivery Rate (multiple mesh sizes)")
    plt.ylim(0.7, 1.02)
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Plot 2: Fault % vs Average Hops ---
    plt.figure(figsize=(8, 5))
    plt.plot(mesh_5["fault_pct"], mesh_5["hops"], marker='o', label='5x5')
    plt.plot(mesh_8["fault_pct"], mesh_8["hops"], marker='o', label='8x8')
    plt.plot(mesh_12["fault_pct"], mesh_12["hops"], marker='o', label='12x12')
    plt.plot(mesh_16["fault_pct"], mesh_16["hops"], marker='o', label='16x16')
    # note: 16x16 hops data not available -> omitted
    plt.xlabel("Fault Percentage (%)")
    plt.ylabel("Average Hops")
    plt.title("Fault % vs Average Hops (5x5, 8x8, 12x12)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Plot 3: Fault % vs Average Reward ---
    plt.figure(figsize=(8, 5))
    plt.plot(mesh_5["fault_pct"], mesh_5["reward"], marker='o', label='5x5')
    plt.plot(mesh_8["fault_pct"], mesh_8["reward"], marker='o', label='8x8')
    plt.plot(mesh_12["fault_pct"], mesh_12["reward"], marker='o', label='12x12')
    plt.plot(mesh_16["fault_pct"], mesh_16["reward"], marker='o', label='16x16')
    # note: 16x16 reward data not available -> omitted
    plt.xlabel("Fault Percentage (%)")
    plt.ylabel("Average Reward")
    plt.title("Fault % vs Average Reward (5x5, 8x8, 12x12)")
    plt.grid(True)
    plt.legend()
    plt.show()


def draw_grid(ax, env, active_packets, show_ports=True):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(0, env.GRID_W)
    ax.set_ylim(0, env.GRID_H)
    ax.invert_yaxis()  # so (0,0) is top-left like your coordinate system
    ax.set_xticks([])
    ax.set_yticks([])

    # draw cells
    for r in range(env.GRID_H):
        for c in range(env.GRID_W):
            node = (r, c)
            # base color
            face = NODE_BG
            if node in env.faulty_nodes:
                face = NODE_FAULTY_BG
            # if any sustained port exists at this node, slightly orange tinted
            node_has_sustained = any(((node, p) in env.sustained_faulty_ports) for p in [0,1,2,3,env.LOCAL_PORT])
            if node_has_sustained and (node not in env.faulty_nodes):
                face = NODE_SUSTAIN_BG

            rect = Rectangle((c, r), 1, 1, facecolor=face, edgecolor=NODE_BORDER)
            ax.add_patch(rect)

    # show port markers (triangles) for sustained ports
    if show_ports:
        print(env.sustained_faulty_ports)
        for ((nr, nc), port) in env.sustained_faulty_ports:
            if not env.in_bounds(nr, nc):
                continue
            # compute triangle position inside cell according to port: 0=UP,1=RIGHT,2=DOWN,3=LEFT
            cx = nc + 0.5
            cy = nr + 0.5
            offset = 0.35
            if port == 3:  # UP - triangle pointing up
                tri = RegularPolygon((cx, cy - offset), numVertices=3, radius=0.15, orientation=0)
            elif port == 0:  # RIGHT
                tri = RegularPolygon((cx + offset, cy), numVertices=3, radius=0.15, orientation=0.5 * np.pi)
            elif port == 1:  # DOWN
                tri = RegularPolygon((cx, cy + offset), numVertices=3, radius=0.15, orientation=np.pi)
            elif port == 2:  # LEFT
                tri = RegularPolygon((cx - offset, cy), numVertices=3, radius=0.15, orientation=1.5 * np.pi)
            else:  # local
                tri = Circle((cx, cy), 0.08)
            tri.set_facecolor("#7f0000")
            tri.set_edgecolor("k")
            ax.add_patch(tri)

    # draw packets
    for i, pkt in enumerate(active_packets):
        px, py = pkt.pos  # (row,col)
        cx = py + 0.5
        cy = px + 0.5
        color = PACKET_COLORS[i % len(PACKET_COLORS)]
        circle = Circle((cx, cy), 0.25, color=color, zorder=4)
        ax.add_patch(circle)
        ax.text(cx, cy, (pkt.dst_idx, pkt.age), color="white", weight="bold", ha="center", va="center", zorder=5, fontsize=8)

    # draw destinations as small black squares (if any active packets)
    for pkt in active_packets:
        dx, dy = env.idx_to_coord(pkt.dst_idx)
        cx = dy + 0.5
        cy = dx + 0.5
        dest_rect = Rectangle((cx - 0.12, cy - 0.12), 0.24, 0.24, facecolor="#000000", zorder=3)
        ax.add_patch(dest_rect)

    ax.set_title(f"Packets: {len(active_packets)} | Sustained ports: {len(env.sustained_faulty_ports)} | Faulty nodes: {len(env.faulty_nodes)}")
    return ax



def extract_table(q_table, index_file):
    # load q-table
    with open(q_table, 'rb') as f:
        qtable = pickle.load(f)
    new_q_table = []
    max_q_table = []
    for index in range(len(qtable)):
        max_q_table.append(int(np.argmax(qtable[index])))
        # new_q_table[(key)] = [int(x) for x in ss.rankdata(value)]
        # with open(rank_file, 'w') as tf:
        #     # Convert the data to a string and write it
        #     tf.write(str(new_q_table))

        with open(index_file, 'w') as tf:
            # Convert the data to a string and write it
            tf.write(str(max_q_table))

# q_table = "qtable_grid5x5_vcs4_1768334084.pkl"
# index_file = "index.txt"
#
# extract_table(q_table, index_file)

# import numpy as np
import ast
import io
import os

def load_indices(path):
    """
    Load an array of integers from:
      - .npy file (numpy)
      - a whitespace or newline separated text file (one int per line)
      - a Python list literal like: [0, 1, 2, ...]
    Returns a 1-D numpy array of dtype int.
    """
    path = str(path)
    if path.endswith('.npy') and os.path.exists(path):
        return np.load(path).astype(int)

    with open(path, 'r') as f:
        txt = f.read().strip()

    # try whitespace/newline separated numbers
    try:
        arr = np.fromstring(txt, sep=' ')
        if arr.size > 0:
            return arr.astype(int)
    except Exception:
        pass

    # try np.loadtxt style (handles one-per-line)
    try:
        arr = np.loadtxt(io.StringIO(txt), dtype=int)
        # np.loadtxt returns scalar for single value
        return np.atleast_1d(arr).astype(int)
    except Exception:
        pass

    # try parsing a Python list literal
    try:
        lst = ast.literal_eval(txt)
        return np.array(lst, dtype=int)
    except Exception:
        raise ValueError(f"Could not parse indices from {path!r}. "
                         "Supported: .npy, newline-separated numbers, or Python list literal.")

def format_qarray(arr, out_file, per_row=10, var_name="Q_array", pad_width=20):
    """
    Write the array into Verilog-like assignments with `per_row` assignments per line.
      - arr: 1D array-like of integers
      - out_file: output filename
      - per_row: number of assignments per row (10 for your request)
      - var_name: variable name to use (Q_array by default)
      - pad_width: minimal width to pad each assignment for alignment
    """
    arr = np.asarray(arr, dtype=int).flatten()
    n = arr.size

    # sanity check
    if n % per_row != 0:
        print(f"Warning: length {n} is not divisible by {per_row}. Last line will have fewer entries.")

    with open(out_file, 'w') as f:
        for base in range(0, n, per_row):
            entries = []
            for i in range(per_row):
                idx = base + i
                if idx >= n:
                    break
                entry = f"{var_name}[{idx}] = {int(arr[idx])};"
                # pad to keep columns aligned
                entries.append(entry.ljust(pad_width))
            line = "  ".join(entries)
            f.write(line + "\n")

    print(f"Wrote {n} assignments ({(n + per_row - 1)//per_row} lines) to {out_file}")

# # if __name__ == "__main__":
# # change these filenames as needed:
# input_file = "index.txt"        # or "indices.npy" or "indices_list.txt"
# output_file = "q_array_verilog.txt"
# per_row = 10
#
# indices = load_indices(input_file)
# print("Loaded indices shape:", indices.shape)
# format_qarray(indices, output_file, per_row=per_row, var_name="Q_array", pad_width=18)
#
# # show first 3 lines for quick check
# with open(output_file, 'r') as f:
#     for _ in range(3):
#         print(f.readline().rstrip())

# plot_graphs()