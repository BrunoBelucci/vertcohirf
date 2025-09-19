import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score

def eval_centers(data, centers):
    k = len(centers)
    if isinstance(data, list):
        data = np.concatenate(data, axis=1)
    print(f"data shape: {data.shape}")
    scores = np.zeros(shape=(data.shape[0], k))
    for i in range(k):
        scores[:, i] = np.linalg.norm(data - centers[i], axis=1)
    score = np.square(np.min(scores, axis=1))
    score = np.sum(score) / data.shape[0]
    return score


def eval_homogeneity_score(data, centers, true_labels):
    k = len(centers)
    if isinstance(data, list):
        whole_data = np.concatenate(data, axis=1)
    else:
        whole_data = data
    if true_labels is None:
        vinilla_kmeans = KMeans(n_clusters=k).fit(whole_data)
        true_labels = vinilla_kmeans.labels_
    scores = np.zeros(shape=(whole_data.shape[0], k))
    for i in range(k):
        scores[:, i] = np.linalg.norm(whole_data - centers[i], axis=1)
    pred_labels = np.argmin(scores, axis=1)
    homogeneity = homogeneity_score(true_labels, pred_labels)
    completeness = completeness_score(true_labels, pred_labels)
    return homogeneity, completeness
