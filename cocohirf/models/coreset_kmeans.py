import numpy as np
from cocohirf.models.distributed_kmeans import DistributedKMeans


class CoresetKMeans(DistributedKMeans):

    def __init__(
        self,
        coreset_size_div=10,
        alpha=2,
        n_jobs=1,
        random_state=None,
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
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol

    # Override because we need other outputs to aggregate data
    def run_local_kmeans(self, X, i_group): # type: ignore
        features = self.features_groups[i_group]
        X_group = X[:, features]
        labels, _, kmeans = super().run_local_kmeans(X, i_group, return_kmeans=True) # type: ignore
        cost = kmeans.inertia_ / X_group.shape[0]
        distances_from_centers = kmeans.transform(X_group)
        # Select the distance from the closest center for each sample
        distances_from_cluster_centers = distances_from_centers[np.arange(X_group.shape[0]), labels]
        return labels, cost, distances_from_cluster_centers

    def aggregate_data(self, results_i, X, features_groups, sample_weight=None):
        labels_i, costs_i, distances_i = zip(*results_i)
        n_agents = len(features_groups)
        n_samples = len(labels_i[0])
        sensitivity = np.zeros((n_samples, n_agents))
        coreset_size = n_samples // self.coreset_size_div
        for i in range(n_agents):
            group_count = np.zeros(self.kmeans_n_clusters)
            group_cost = np.zeros(self.kmeans_n_clusters)
            group_count = np.bincount(labels_i[i], minlength=self.kmeans_n_clusters)
            group_cost = np.bincount(labels_i[i], weights=distances_i[i] ** 2, minlength=self.kmeans_n_clusters)
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
                where=denom1 != 0,
            )
            term2 = np.divide(
                2 * self.alpha * group_cost[labels_i[i]],
                denom2,
                out=np.zeros_like(distances_i[i], dtype=float),
                where=denom2 != 0,
            )
            term3 = np.divide(4 * n_samples, denom3, out=np.zeros_like(distances_i[i], dtype=float), where=denom3 != 0)
            sensitivity[:, i] = term1 + term2 + term3

        sensitivity_sum = np.sum(sensitivity, axis=1)
        weights = sensitivity_sum / sensitivity_sum.sum()
        indices = self.random_state.choice(len(sensitivity_sum), size=coreset_size, replace=False, p=weights)
        data_cat = np.hstack((X[indices], (1 / sensitivity_sum[indices]).reshape(-1, 1)))
        server_X = data_cat[:, :-1]
        server_weight = data_cat[:, -1]
        server_weight = server_weight / np.sum(server_weight)
        return server_X, server_weight
    