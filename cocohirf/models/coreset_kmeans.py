import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import pandas as pd
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


class CoresetKMeans(ClusterMixin, BaseEstimator):

    def __init__(
        self,
        coreset_size_div=10,
        alpha=2,
        n_jobs=1,
        random_state=None,
        corrupted_agents=None,
        # kmeans parameters
        kmeans_n_clusters: int = 8,
        kmeans_init: str = "k-means++",
        kmeans_n_init: str | int = "auto",
        kmeans_max_iter: int = 300,
        kmeans_tol: float = 1e-4,
    ):
        self.coreset_size_div = coreset_size_div
        self.alpha = alpha
        self.n_jobs = n_jobs
        self._random_state = random_state
        self.corrupted_agents = corrupted_agents
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol

    @property
    def random_state(self):
        if self._random_state is None:
            self._random_state = np.random.default_rng()
        elif isinstance(self._random_state, int):
            self._random_state = np.random.default_rng(self._random_state)
        return self._random_state

    def run_local_kmeans(self, X, i_group):
        child_random_state = np.random.default_rng([self.random_state.integers(0, int(1e6)), i_group])
        random_state = child_random_state.integers(0, 1e6)
        kmeans = KMeans(
            n_clusters=self.kmeans_n_clusters,
            init=self.kmeans_init,
            n_init=self.kmeans_n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.kmeans_tol,
            random_state=random_state,
        )
        features = self.features_groups[i_group]
        X_group = X[:, features]
        labels = kmeans.fit_predict(X_group)
        cost = kmeans.inertia_ / X_group.shape[0]
        distances_from_centers = kmeans.transform(X_group)
        # Select the distance from the closest center for each sample
        distances_from_cluster_centers = distances_from_centers[np.arange(X_group.shape[0]), labels]
        return labels, cost, distances_from_cluster_centers

    def fit( # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        self.features_groups = features_groups
        n_agents = len(features_groups)
        n_samples = X.shape[0]
        coreset_size = n_samples // self.coreset_size_div 
        results_i = Parallel(n_jobs=self.n_jobs)(delayed(self.run_local_kmeans)(X, r) for r in range(n_agents))
        labels_i, costs_i, distances_i = zip(*results_i)
        sensitivity = np.zeros((n_samples, n_agents))
        for i in range(n_agents):
            group_count = np.zeros(self.kmeans_n_clusters)
            group_cost = np.zeros(self.kmeans_n_clusters)
            group_count = np.bincount(labels_i[i], minlength=self.kmeans_n_clusters)
            group_cost = np.bincount(labels_i[i], weights=distances_i[i]**2, minlength=self.kmeans_n_clusters)
            # Safe division to avoid division by zero

            # Precompute denominators
            denom1 = costs_i[i]
            denom2 = group_count[labels_i[i]] * costs_i[i]
            denom3 = group_count[labels_i[i]]

            # Safe division for each term
            term1 = np.divide(
                self.alpha * (distances_i[i] ** 2),
                denom1,
                out=np.zeros_like(distances_i[i], dtype=float),
                where=denom1 != 0
            )
            term2 = np.divide(
                2 * self.alpha * group_cost[labels_i[i]],
                denom2,
                out=np.zeros_like(distances_i[i], dtype=float),
                where=denom2 != 0
            )
            term3 = np.divide(
                4 * n_samples,
                denom3,
                out=np.zeros_like(distances_i[i], dtype=float),
                where=denom3 != 0
            )
            sensitivity[:, i] = term1 + term2 + term3

        sensitivity_sum = np.sum(sensitivity, axis=1)
        weights = sensitivity_sum / sensitivity_sum.sum()
        indices = self.random_state.choice(len(sensitivity_sum), size=coreset_size, replace=False, p=weights)
        C = np.hstack((X[indices], (1 / sensitivity_sum[indices]).reshape(-1, 1)))
        data = C[:, :-1]
        weights = C[:, -1]
        weights = weights / np.sum(weights)
        server_kmeans = KMeans(
            n_clusters=self.kmeans_n_clusters,
            init=self.kmeans_init,
            n_init=self.kmeans_n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.kmeans_tol,
            random_state=42,
        )
        server_kmeans.fit(data, sample_weight=weights)
        server_kmeans_clusters_centers = server_kmeans.cluster_centers_
        labels = []
        costs = []
        for i in range(n_agents):
            X_group = X[:, features_groups[i]]
            dist = [
                np.linalg.norm(X_group - server_kmeans_clusters_centers[j][features_groups[i]], axis=1)
                for j in range(self.kmeans_n_clusters)
            ]
            dist = np.asarray(dist).T
            closest_dist = np.min(dist, axis=1)
            cost = np.sum(closest_dist**2) / n_samples
            labels.append(np.argmin(dist, axis=1))
            costs.append(cost)
        self.labels_ = labels
        self.costs_ = costs

    def fit_predict( # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        self.fit(X, features_groups, y, sample_weight)
        return self.labels_
