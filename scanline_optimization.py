import numpy as np

def delta(x, y):
    return abs(x - y)

def regularisation_cost(dx, prev):  # returns a V value
    x = delta(dx, prev)
    if x == 1: return 0
    elif dx == prev: return 1
    else: return x - 1

def generalized_scanline_optimization(D, lmbd):
    """
    Alg 1: dynamic programming scanline optimization
    """
    rows, cols = D.shape
    
    # initialize H with zeros 
    H = np.zeros((rows, cols))

    # set the first row (base case) 
    H[0, :] = D[0, :]

    h_vals = np.zeros((rows, cols))
    # iterate through the first sequence (rows)
    for x in range(1, rows):
        # solve min(H[x-1, d'] + lambda*V(d, d')) for all d
        for d in range(cols):
            hh_vals = []
            for d_p in range(cols):
                hh_vals.append(H[x-1, d_p] + lmbd*regularisation_cost(d,d_p))
            h_vals[x][d] = min(hh_vals)

        # calculate the current row's costs
        for d in range(cols):
            # final calculation: dissimilarity + best previous path
            # the min(0, ...) resets the score if the path becomes bad
            H[x, d] = min(0, D[x][d] + h_vals[x][d])

    return H


import numpy as np

def backtrack_path(H, lmbda):
    """
    H: The accumulated cost matrix (n x m)
    lmbda: The regularization weight used during matrix calculation
    """
    n, m = H.shape

    # find min value of array, check whole array just to be sure
    flat_idx = np.argmin(H)
    current_x_raw, current_d_raw = np.unravel_index(flat_idx, H.shape)

    current_x = int(current_x_raw)
    current_d = int(current_d_raw)

    path = [(current_x, current_d)]

    # 2. Iterate backwards from the end of sequence 1 to the start
    while current_x > 0:
        # if score is too bad we stop the path like in the paper
        if H[current_x, current_d] >= 0:
            break
        prev_x = current_x - 1
        best_prev_d = -1
        min_prev_cost = float('inf')

        # Search across possible previous d' values
        for prev_d in range(m):
            v_cost = regularisation_cost(dx=current_d, prev=prev_d)

            # Reconstruct the score from the previous step
            total_prev_cost = H[prev_x, prev_d] + (lmbda * v_cost)

            if total_prev_cost < min_prev_cost:
                min_prev_cost = total_prev_cost
                best_prev_d = prev_d

        current_d = best_prev_d
        current_x = prev_x
        path.append((current_x, current_d))

    # The path is gathered backwards, so reverse it
    return path[::-1]
