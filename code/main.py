# from vsm import create_vsm, compute_vsm
from scanline_optimization import generalized_scanline_optimization as scanl_opt
from scanline_optimization import backtrack_path, normalized_score
import numpy as np
import util

def main():
    # S = compute_vsm("sequence1/", "sequence2/")
    # write_matrix_to_csv(S, "vsm")
    S = util.read_from_csv("vsm")
    D = - S
    lbd = 1.2
    H = scanl_opt(D, lbd)
    path = backtrack_path(H, lbd)
    util.plot_matrix(D, "Distance Matrix", "Cosine Similarity")
    util.plot_matrix(H, "H matrix", "")
    Z = np.zeros_like(D)
    P = util.path_matrix(path, Z)
    util.plot_matrix(P, "PATH", "")

    N = normalized_score(H, path)
    util.plot_matrix(N, "Normalized Scores", "")

main()
