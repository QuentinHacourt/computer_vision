from vsm import create_vsm
from scanline_optimization import generalized_scanline_optimization as scanl_opt
from scanline_optimization import backtrack_path
import matplotlib.pyplot as plt
import numpy as np

def main():
    S = create_vsm("images/sequence1/", "images/sequence2/")
    D = -S
    H = scanl_opt(D, 1.2)
    path = backtrack_path(H, 1.2)
    plot_matrix(D, "Distance Matrix", "Euclidean Similarity")
    plot_matrix(H, "H matrix", "")
    print(f"Path length: {len(path)}")
    print(path)
    Z = np.zeros_like(D)
    P = path_matrix(path, Z)
    plot_matrix(P, "PATH", "")


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

# def compute_vsm(image_paths):
#     # Initialize a RootSIFT feature extractor
#     feature_extractor = SIFT()
#     # Initialize a Fisher Vector encoder
#     fisher_encoder = FisherVectorEncoder(feature_extractor=feature_extractor)

#     # Train the encoder with GMM clustering
#     fisher_encoder.learn(image_paths, n_clusters=10)

#     # Flatten features from images
#     features = np.vstack([feature_extractor(image) for image in images])

#     if os.path.exists("pca_model.pkl"):
#             print("--- Load PCA --- ")
#             pca = joblib.load("pca_model.pkl")
#     else:
#         # Train PCA
#         output_dim = feature_extractor.output_dim
#         pca = PCA(n_components=output_dim // 2)

#     pca.fit(features)

#     # Save PCA model
#     joblib.dump(pca, "pca_model.pkl")
#     features_transformed = pca.transform(features)

#     if os.path.exists("gmm_model.pkl"):
#         print("--- Load GMM --- ")
#         gmm = joblib.load("gmm_model.pkl")
#     else:
#         gmm = GaussianMixture(n_components=10, covariance_type='diag')
#         gmm.fit(features_transformed)

#     # Save GMM model
#     joblib.dump(gmm, "gmm_model.pkl")


#     # Initialize Fisher Vector encoder with pre-trained GMM
#     fisher_encoder = FisherVectorEncoder(feature_extractor=feature_extractor, gmm_model=gmm, pca=pca)

#     print("----- Building Matrix --------")
#     # Build VSM
#     matrix = np.zeros((len(image_paths), len(image_paths)))

#     for i in range(len(image_paths)):
#         for j in range(i, len(image_paths)):
#             value = fisher_encoder.similarity_score(image_paths[i], image_paths[j])
#             matrix[i][j] = value
#             if i != j:
#                 matrix[j][i] = value

#     # TEST
#     image_1, image_2 = images[0], images[1]
#     plot_image(image_1, title="Image 1")
#     plot_image(image_2, title="Image 2")

#     # Compute similarity using Fisher Vector encoder
#     similarity = fisher_encoder.similarity_score(image_1, image_2)

#     print("Similarity score:", similarity)

#     image_3, image_4 = images[0], images[19]
#     plot_image(image_3, title="Image 1")
#     plot_image(image_4, title="Image 20")

#     similarity_2 = fisher_encoder.similarity_score(image_3, image_4)

#     print("Similarity score:", similarity_2)
#     return matrix


# print("------- Extract SIFT Features -------")

# path = "images/"
# try:
#     image_paths = sorted(glob.glob("images/*.JPG"))
#     images = []
#     for path in image_paths:
#         images.append(cv.imread(path, cv.IMREAD_COLOR))
# except:
#     raise Exception("Images folder does not exist, or folder is empty")

# print(len(image_paths))

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
