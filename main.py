from vsm import create_vsm, compute_vsm
from scanline_optimization import generalized_scanline_optimization as scanl_opt
from scanline_optimization import backtrack_path
import matplotlib.pyplot as plt
import numpy as np

def main():
    #S = compute_vsm("images/sequence1/", "images/sequence2/")
    #write_matrix_to_csv(S, "vsm")
    S = read_from_csv("vsm")
    D = S - 1
    try_lambdas(D)
    # D = -S
    # H = scanl_opt(D, 1.2)
    # path = backtrack_path(H, 1.2)
    # plot_matrix(D, "Distance Matrix", "Euclidean Similarity")
    # plot_matrix(H, "H matrix", "")
    # # print(f"Path length: {len(path)}")
    # # print(path)
    # Z = np.zeros_like(D)
    # P = path_matrix(path, Z)
    # plot_matrix(P, "PATH", "")

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









# import joblib
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import euclidean_distances
# import glob
# import os
# import warnings
# warnings.filterwarnings('ignore')

# from sklearn.mixture import GaussianMixture
# import scanline_optimization as scan_opt
# #--------
# from pyvisim.encoders import FisherVectorEncoder
# from pyvisim.features import SIFT
# from pyvisim._utils import plot_image

# # ------ old
# def extract_descriptors(image_paths):
#     sift = cv.SIFT_create()
#     all_descriptors = []
#     img_descriptor_map = {}

#     for path in image_paths:
#         img = cv.imread(path, cv.IMREAD_GRAYSCALE)
#         if img is None:
#             continue

#         keypoints, descriptors = sift.detectAndCompute(img, None)
#         if descriptors is not None:
#             all_descriptors.append(descriptors)
#             img_descriptor_map[path] = descriptors
#         else:
#             img_descriptor_map[path] = np.array([])

#     return np.vstack(all_descriptors), img_descriptor_map

# def compute_bow(descriptor_map, kmeans_model):
#     image_features = []

#     vocab_size = kmeans_model.n_clusters

#     for path, descriptors in descriptor_map.items():
#         histogram = np.zeros(vocab_size)
#         if descriptors.size > 0:
#             visual_words = kmeans_model.predict(descriptors)
#             for i in visual_words:
#                 histogram[i] += 1

#         histogram /= np.sum(histogram) if np.sum(histogram) != 0 else 1
#         image_features.append(histogram)

#     return np.array(image_features)
# #------------------

# #------------------


# print("------- Extract SIFT Features -------")

# print("--------- Compute VSM --------")
# if os.path.exists("distance_matrix.csv"):
#     print("--- Load Distance Matrix --- ")
#     distance_matrix = pd.read_csv('distance_matrix.csv', header=None)

# else:
#     distance_matrix = compute_vsm(images)
# np.savetxt("distance_matrix.csv", distance_matrix)

# descriptors, descriptor_map = extract_descriptors(image_paths)

# kmeans = KMeans(10, random_state=42)
# kmeans.fit(descriptors)

# print("Computing Bag of Visual Words")
# bovw_matrix = compute_bow(descriptor_map, kmeans)

# print("distance matrix")
# distance_matrix = euclidean_distances(bovw_matrix, bovw_matrix)
# print(distance_matrix)

# fig, ax = plt.subplots()
# im = ax.imshow(distance_matrix)

# cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")

# ax.set_title("Distance Matrix")
# fig.tight_layout()
# plt.show()

# # #-------------------
# H = scan_opt.generalized_scanline_optimization(-distance_matrix, 0.5)
# print(H)
# #np.savetxt("H_matrix.csv", H, delimiter=",")

# # # ------------
# fig, ax = plt.subplots()
# im = ax.imshow(H)
# ax.set_title("H Matrix")
# fig.tight_layout()
# plt.show()
