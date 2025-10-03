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
        use_server_labels: bool = True,  # if True, use the server labels, if False use local labels of each agent
    ):
        self.n_jobs = n_jobs
        self._random_state = random_state
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.use_server_labels = use_server_labels

    @property
    def random_state(self):
        if self._random_state is None:
            self._random_state = np.random.default_rng()
        elif isinstance(self._random_state, int):
            self._random_state = np.random.default_rng(self._random_state)
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        if value is None:
            self._random_state = np.random.default_rng()
        elif isinstance(value, int):
            self._random_state = np.random.default_rng(value)
        else:
            raise ValueError("random_state must be an integer or None.")

    def run_local_kmeans(self, X, i_group, return_kmeans=False):
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
        # Select the distance from the closest center for each sample
        cluster_centers = kmeans.cluster_centers_
        if return_kmeans:
            return labels, cluster_centers, kmeans
        return labels, cluster_centers

    def aggregate_data(self, results_i, X, features_groups, sample_weight=None):
        labels_i, cluster_centers_i = zip(*results_i)
        n_agents = len(features_groups)
        n_samples = len(labels_i[0])
        sample_weight = np.ones(n_samples) / n_samples if sample_weight is None else sample_weight

        # Vectorized center_list construction
        # Get all possible combinations of cluster indices for n_agents
        all_indices = np.array(np.meshgrid(*[np.arange(self.kmeans_n_clusters)] * n_agents)).T.reshape(-1, n_agents)
        # THIS IS EQUIVALENT but marginally faster than using itertools.product
        # all_indices = np.array(list(product(range(self.kmeans_n_clusters), repeat=n_agents)))
        # For each agent, select the cluster centers for all combinations
        center_parts = [cluster_centers_i[j][all_indices[:, j], :] for j in range(n_agents)]
        # Concatenate along feature axis
        server_X = np.concatenate(center_parts, axis=1)

        # indices: map each sample to a unique center in server_X (the index in server_X)
        labels_matrix = np.vstack(labels_i)
        indices = np.sum(labels_matrix[::-1, :] * (self.kmeans_n_clusters ** np.arange(n_agents)[:, None]), axis=0)
        # Use bincount to sum sample_weight for each unique index
        server_weight = np.bincount(indices, weights=sample_weight, minlength=server_X.shape[0])
        return server_X, server_weight, indices

    def fit( # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        # we will work with numpy (arrays) for speed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.features_groups = features_groups
        n_agents = len(features_groups)
        results_i = Parallel(n_jobs=self.n_jobs)(delayed(self.run_local_kmeans)(X, r) for r in range(n_agents))
        server_X, server_weight, indices = self.aggregate_data(results_i, X, features_groups, sample_weight)
        server_n_clusters = min(self.kmeans_n_clusters, server_X.shape[0])
        server_kmeans = KMeans(
            n_clusters=server_n_clusters,
            init=self.kmeans_init,
            n_init=self.kmeans_n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.kmeans_tol,
            random_state=self.random_state.integers(0, 1e6),
        )
        server_centers_labels = server_kmeans.fit_predict(server_X, sample_weight=server_weight)
        if self.use_server_labels:
            # we use the label corresponding to the center assigned to each sample
            labels = server_centers_labels[indices]
        else:
            # each agent finds the closest center using its own features
            server_kmeans_clusters_centers = server_kmeans.cluster_centers_
            labels = []
            for i in range(n_agents):
                X_group = X[:, features_groups[i]]
                dist = [
                    np.linalg.norm(X_group - server_kmeans_clusters_centers[j][features_groups[i]], axis=1)
                    for j in range(server_n_clusters)
                ]
                dist = np.asarray(dist).T
                labels.append(np.argmin(dist, axis=1))
        self.labels_ = labels
        return self

    def fit_predict(  # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        self.fit(X, features_groups, y, sample_weight)
        return self.labels_
