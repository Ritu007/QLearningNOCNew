# from sys import dllhandle
#
# from inference import destination

# neigh = [1, 1, 1, 1, 1, 1, 1, 1]
# dir = 3
#
# state = dir
# for i in range(0, 8):
#     state = state * 2 + neigh[i]
#
# print(state)


def get_state(source, destination, faulty_routers):
    sx, sy = source
    dx, dy = destination
    delx = dx - sx
    dely = dy - sy

    pos = ["RD", "DL", "LU", "UR"]

    dir = 0

    if dely > 0:
        if delx >= 0:
            dir = 0
        else:
            dir = 3
    elif dely == 0:
        if delx > 0:
            dir = 1
        else:
            dir = 3
    else:
        if delx > 0:
            dir = 1
        else:
            dir = 2

    print("dir", dir)
    delta = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    delta_th = [[(0, 2), (1, 1), (2, 0), (1, 2)], [(2, 0), (1, -1), (0, -2), (2, -1)], [(0, -2), (-1, -1), (-2, 0), (-1, -2)],[(-2, 0), (-1, 1), (0, 2), (-2, 1)]]
    neigh = []

    for i in range(0, 4):
        neigh.append(1 if (sx + delta[i][0], sy + delta[i][1]) in faulty_routers else 0)

    for i in range(0, 4):
        neigh.append(1 if (sx + delta_th[dir][i][0], sy + delta_th[dir][i][1]) in faulty_routers else 0)

    print("neigh", neigh)

    state = dir

    for i in range(0, 8):
        state = state * 2 + neigh[i]

    print("state", state)
    return state


def get_state_new(source, destination, faulty_routers, N=None):
    # source/destination: (row, col)
    sx, sy = source
    dx, dy = destination
    delx = dx - sx
    dely = dy - sy

    # dir mapping: 0=right,1=down,2=left,3=up
    if dely > 0:
        dir = 0 if delx >= 0 else 3
    elif dely == 0:
        dir = 1 if delx > 0 else 3
    else:  # dely < 0
        dir = 1 if delx > 0 else 2
    print("dir", dir)
    # neighbors: right, down, left, up
    deltas = [(0,1),(1,0),(0,-1),(-1,0)]
    delta_th = [
        [(0,2),(1,1),(2,0),(1,2)],
        [(2,0),(1,-1),(0,-2),(2,-1)],
        [(0,-2),(-1,-1),(-2,0),(-1,-2)],
        [(-2,0),(-1,1),(0,2),(-2,1)]
    ]

    faulty_set = set(faulty_routers)
    def is_fault(p):
        if N is not None:
            r,c = p
            if not (0 <= r < N and 0 <= c < N):
                return 1
        return 1 if p in faulty_set else 0
    neigh = []
    bits = 0
    for off in deltas:
        bits = (bits << 1) | is_fault((sx + off[0], sy + off[1]))
        neigh.append(is_fault((sx + off[0], sy + off[1])))
    for off in delta_th[dir]:
        bits = (bits << 1) | is_fault((sx + off[0], sy + off[1]))
        neigh.append(is_fault((sx + off[0], sy + off[1])))
    print("NEigh", neigh)
    packed = (dir << 8) | bits
    # optional: return components for debugging
    return packed  # or return dir, bits, packed


source = (2, 3)
destination = (3, 4)

faulty_routers = [(3, 3), (2, 3), (3, 4)]

state = get_state_new(source, destination, faulty_routers, N=5)
print(state)