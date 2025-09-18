import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import pandas as pd
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


class DistributedKMeans(ClusterMixin, BaseEstimator):

    def __init__(
        self,
        n_jobs=1,
        random_state=None,
        # kmeans parameters
        kmeans_n_clusters: int = 8,
        kmeans_init: str = "k-means++",
        kmeans_n_init: str | int = "auto",
        kmeans_max_iter: int = 300,
        kmeans_tol: float = 1e-4,
    ):
        self.n_jobs = n_jobs
        self._random_state = random_state
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
        cluster_centers = kmeans.cluster_centers_
        return labels, cost, distances_from_cluster_centers, cluster_centers

    def fit( # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples) / n_samples if sample_weight is None else sample_weight
        self.features_groups = features_groups
        n_agents = len(features_groups)
        results_i = Parallel(n_jobs=self.n_jobs)(delayed(self.run_local_kmeans)(X, r) for r in range(n_agents))
        labels_i, costs_i, distances_i, cluster_centers_i = zip(*results_i)

        # Vectorized center_list construction
        # Get all possible combinations of cluster indices for n_agents
        all_indices = np.array(np.meshgrid(*[np.arange(self.kmeans_n_clusters)] * n_agents)).T.reshape(-1, n_agents)
        # For each agent, select the cluster centers for all combinations
        center_parts = [
            cluster_centers_i[j][all_indices[:, j], :] for j in range(n_agents)
        ]
        # Concatenate along feature axis
        center_list = np.concatenate(center_parts, axis=1)

        labels_matrix = np.vstack(labels_i)
        indices = np.sum(labels_matrix[::-1, :] * (self.kmeans_n_clusters ** np.arange(n_agents)[:, None]), axis=0)
        # Use bincount to sum sample_weight for each unique index
        center_weights = np.bincount(indices, weights=sample_weight, minlength=center_list.shape[0])

        server_kmeans = KMeans(
            n_clusters=self.kmeans_n_clusters,
            init=self.kmeans_init,
            n_init=self.kmeans_n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.kmeans_tol,
            random_state=self.random_state.integers(0, 1e6),
        )
        server_kmeans.fit(center_list, sample_weight=center_weights)
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

    def fit_predict(  # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        self.fit(X, features_groups, y, sample_weight)
        return self.labels_
