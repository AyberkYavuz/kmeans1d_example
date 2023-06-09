import random
import kmeans1d
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin


def generate_items(n_items: int) -> list:
    items = list()
    for i in range(0, n_items):
        n = random.randint(1, 1000)
        items.append(n)
    return items


def predict_clusters_version1(centroids: list) -> list:
    clusters_for_test_feture = []
    for x in test_feature:
        distances = [abs(x - c) for c in centroids]
        clusters_for_test_feture.append(distances.index(min(distances)))
    return clusters_for_test_feture


def predict_clusters_version2(test_feature: list, centroids: list) -> np.ndarray:
    test_feature = np.array(test_feature)[:, np.newaxis]
    centroids = np.array(centroids)[:, np.newaxis]

    # produce cluster ids with numpy
    clusters_np = np.argmin(cdist(test_feature, centroids), axis=1)

    return clusters_np


def predict_clusters_version3(test_feature: list, centroids: list) -> np.ndarray:
    test_feature = np.array(test_feature)[:, np.newaxis]
    centroids = np.array(centroids)[:, np.newaxis]

    # produce cluster ids with sklearn
    clusters_sklearn = pairwise_distances_argmin(test_feature, centroids)
    return clusters_sklearn


training_feature = generate_items(10000)

k = 10

clusters, centroids = kmeans1d.cluster(training_feature, k)


test_feature = generate_items(2000)

# runs slower
# produce cluster ids for test_feature
clusters_for_test_feture = predict_clusters_version1(centroids)

# runs faster
clusters_np = predict_clusters_version2(test_feature, centroids)

# runs faster
clusters_sklearn = predict_clusters_version3(test_feature, centroids)

