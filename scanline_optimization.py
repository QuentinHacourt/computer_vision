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
