# from vsm import create_vsm, compute_vsm
from scanline_optimization import generalized_scanline_optimization as scanl_opt
from scanline_optimization import backtrack_path, normalized_score
import matplotlib.pyplot as plt
import numpy as np

def main():
    # S = compute_vsm("images/sequence1/", "images/sequence2/")
    # write_matrix_to_csv(S, "vsm")
    S = read_from_csv("vsm")
    D = - S
    lbd = 1.2
    H = scanl_opt(D, lbd)
    path = backtrack_path(H, lbd)
    plot_matrix(D, "Distance Matrix", "Cosine Similarity")
    plot_matrix(H, "H matrix", "")
    Z = np.zeros_like(D)
    P = path_matrix(path, Z)
    plot_matrix(P, "PATH", "")

    N = normalized_score(H, path)
    plot_matrix(N, "Normalized Scores", "")

def try_lambdas(D):
    lambdas = [0.0, 0.5, 1, 2, 5, 10, 50, 100, 1000]

    for lmd in lambdas:
        H = scanl_opt(D, lmd)
        plot_matrix(H, f"lambda = {lmd}","")
        path = backtrack_path(H, lmd)
        Z = np.zeros_like(D)
        P = path_matrix(path, Z)
        plot_matrix(P, f"PATH with lmd = {lmd}", "")

def write_matrix_to_csv(matrix, filename):
    np.savetxt(filename + ".csv", matrix)

def read_from_csv(filename):
    matrix = np.loadtxt(open(filename + ".csv", "r"), delimiter=" ")
    return matrix

def plot_matrix(A, title, y_label):
    fig, ax = plt.subplots()
    im = ax.imshow(A)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(y_label, rotation=-90, va="bottom")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def path_matrix(path, P):
    for x, d in path:
        P[x, d] = 1

    return P

main()
