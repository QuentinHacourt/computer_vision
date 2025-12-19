import numpy as np

def delta(x, y):
    return abs(x - y)

def regularisation_cost(dx, prev):
    x = delta(dx, prev)
    if x == 1: return 0
    elif dx == prev: return 1
    else: return x - 1

def generalized_scanline_optimization(D, lmbd):
    rows, cols = D.shape

    H = np.zeros((rows, cols))

    H[0, :] = D[0, :]

    h_vals = np.zeros((rows, cols))
    for x in range(1, rows):
        for d in range(cols):
            hh_vals = []
            for d_p in range(cols):
                hh_vals.append(H[x-1, d_p] + lmbd*regularisation_cost(d,d_p))
            h_vals[x][d] = min(hh_vals)

        for d in range(cols):
            H[x, d] = min(0, D[x][d] + h_vals[x][d])

    return H

import numpy as np

def backtrack_path(H, lmbda):
    n, m = H.shape

    flat_idx = np.argmin(H)
    current_x_raw, current_d_raw = np.unravel_index(flat_idx, H.shape)

    current_x = int(current_x_raw)
    current_d = int(current_d_raw)

    path = [(current_x, current_d)]

    while current_x > 0:
        if H[current_x, current_d] >= -6.5:
            break
        prev_x = current_x - 1
        best_prev_d = -1
        min_prev_cost = float('inf')

        for prev_d in range(m):
            v_cost = regularisation_cost(dx=current_d, prev=prev_d)

            total_prev_cost = H[prev_x, prev_d] + (lmbda * v_cost)

            if total_prev_cost < min_prev_cost:
                min_prev_cost = total_prev_cost
                best_prev_d = prev_d

        current_d = best_prev_d
        current_x = prev_x
        path.append((current_x, current_d))

    return path[::-1]

def normalized_score(H, path):
    rows, cols = H.shape
    N = np.zeros_like(H)
    denominator = len(path)**2

    for i in range(rows):
        for j in range (cols):
            xs, ds = count_xs_and_ds(path, i, j)
            numerator = H[i,j] * xs * ds
            N[i,j] = numerator / denominator

    return N

def count_xs_and_ds(path, x, d):
    xs = 0
    ds = 0
    for (i, j) in path:
        if i == x:
            xs += 1
        if j == d:
            ds += 1

    return xs, ds
