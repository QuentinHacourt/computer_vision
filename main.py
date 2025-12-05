import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import glob

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


print("Extract SIFT Features")

image_paths = sorted(glob.glob("images/*.JPG"))
descriptors, descriptor_map = extract_descriptors(image_paths)

kmeans = KMeans(10, random_state=42)
kmeans.fit(descriptors)

print("Computing Bag of Visual Words")
bovw_matrix = compute_bow(descriptor_map, kmeans)

print("distance matrix")
distance_matrix = euclidean_distances(bovw_matrix, bovw_matrix)
print(distance_matrix)

fig, ax = plt.subplots()
im = ax.imshow(distance_matrix)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Euclidean Similarity", rotation=-90, va="bottom")

ax.set_title("Distance Matrix")
fig.tight_layout()
plt.show()