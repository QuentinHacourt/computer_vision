import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import glob
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import os
from pyvisim.encoders import FisherVectorEncoder
from pyvisim.features import SIFT
from pyvisim._utils import plot_image
from sklearn.mixture import GaussianMixture
import multiprocessing as mp

def extract_descriptors(image_paths):
    sift = cv.SIFT_create()
    all_descriptors = []
    img_descriptor_map = {}

    for path in image_paths:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue

        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
            img_descriptor_map[path] = descriptors
        else:
            img_descriptor_map[path] = np.array([])

    return np.vstack(all_descriptors), img_descriptor_map

def compute_bow(descriptor_map, kmeans_model):
    image_features = []

    vocab_size = kmeans_model.n_clusters

    for path, descriptors in descriptor_map.items():
        histogram = np.zeros(vocab_size)
        if descriptors.size > 0:
            visual_words = kmeans_model.predict(descriptors)
            for i in visual_words:
                histogram[i] += 1

        histogram /= np.sum(histogram) if np.sum(histogram) != 0 else 1
        image_features.append(histogram)

    return np.array(image_features)

def extract_sift_features(path):
    paths = sorted(glob.glob(path + "*.JPG"))
    descriptors, descriptor_map = extract_descriptors(paths)

    kmeans = KMeans(10, random_state=42)
    kmeans.fit(descriptors)

    bowv = compute_bow(
        descriptor_map=descriptor_map,
        kmeans_model=kmeans
    )

    return bowv

def create_vsm(path1, path2):
    bovw1 = extract_sift_features(path1)
    bovw2 = extract_sift_features(path2)
    distance_matrix = euclidean_distances(bovw1, bovw2)
    return distance_matrix

def compute_vsm(path1, path2):
    try:
        images_path = sorted(glob.glob(path1 + "*.JPG") + glob.glob(path2 + "*.JPG"))
        images_path = list({os.path.basename(p): p for p in images_path}.values())
        print(images_path)
        images = []
        for path in images_path:
            images.append(cv.imread(path, cv.IMREAD_COLOR))
    except:
        raise Exception("Images folder does not exist, or folder is empty")
    
    imagespath1 = sorted(glob.glob(path1 + "*.JPG"))
    imagespath2 = sorted(glob.glob(path2 + "*.JPG"))

    images1 = []
    images2 = []

    print(len(imagespath2))
    for path in imagespath1:
            images1.append(cv.imread(path, cv.IMREAD_COLOR))
    for path in imagespath2:
            images2.append(cv.imread(path, cv.IMREAD_COLOR))


    feature_extractor = SIFT()
    
    if os.path.exists("pca_model_2.pkl"):
            print("--- Load PCA --- ")
            pca = joblib.load("pca_model_2.pkl")
    else:
        # Train PCA
        output_dim = feature_extractor.output_dim
        pca = PCA(n_components=output_dim)
        joblib.dump(pca, "pca_model_2.pkl")

    # Flatten features from images
    features = np.vstack([feature_extractor(image) for image in images])

    pca.fit(features)

    features_transformed = pca.transform(features)

    if os.path.exists("gmm_model_2.pkl"):
        print("--- Load GMM --- ")
        gmm = joblib.load("gmm_model_2.pkl")
    else:
        gmm = GaussianMixture(n_components=10, covariance_type='diag')
        gmm.fit(features_transformed)
        joblib.dump(gmm, "gmm_model_2.pkl")
    
    fisher_encoder = FisherVectorEncoder(feature_extractor=feature_extractor, gmm_model=gmm, pca=pca)
    #fisher_encoder.learn(images, n_clusters=10)

    matrix = np.zeros((len(imagespath1), len(imagespath2)))

    for i in range(len(imagespath1)):
        for j in range(len(imagespath2)):
            value = fisher_encoder.similarity_score(images1[i], images2[j])
            matrix[i][j] = value


    return matrix