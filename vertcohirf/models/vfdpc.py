import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from typing import Literal, cast


class VFDPC(ClusterMixin, BaseEstimator):
    """
    Simplified VFDPC implementation.

    This variant keeps the clustering pipeline from the paper and drops
    secure aggregation details (encryption, random noise matrices, and
    sequential encrypted passing). Distance components from each client
    are merged directly in cleartext.
    """

    def __init__(
        self,
        random_state=None,
        n_jobs: int = 1,
        # target number of final clusters
        kmeans_n_clusters: int = 8,
        # MKNN / local density parameter
        knn_k: int = 10,
        # spectral clustering parameters
        spectral_assign_labels: Literal["kmeans", "discretize", "cluster_qr"] = "cluster_qr",
        spectral_n_init: int = 10,
        # output style compatibility with other methods in this repository
        use_server_labels: bool = True,
        # preprocessing used in the VFDPC paper
        use_minmax_normalization: bool = True,
    ):
        self._random_state = random_state
        self.n_jobs = n_jobs
        self.kmeans_n_clusters = kmeans_n_clusters
        self.knn_k = knn_k
        self.spectral_assign_labels = spectral_assign_labels
        self.spectral_n_init = spectral_n_init
        self.use_server_labels = use_server_labels
        self.use_minmax_normalization = use_minmax_normalization

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

    def _validate_inputs(self, X: np.ndarray, features_groups: list[list[int]]):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array-like object.")
        if len(features_groups) == 0:
            raise ValueError("features_groups must contain at least one client feature group.")
        n_features = X.shape[1]
        for i, group in enumerate(features_groups):
            if len(group) == 0:
                raise ValueError(f"features_groups[{i}] cannot be empty.")
            invalid = [f for f in group if f < 0 or f >= n_features]
            if invalid:
                raise ValueError(f"features_groups[{i}] contains invalid feature indices: {invalid}")

    def _minmax_per_client(self, X: np.ndarray, features_groups: list[list[int]]):
        if not self.use_minmax_normalization:
            return X.copy()
        X_norm = X.copy().astype(float)
        for group in features_groups:
            scaler = MinMaxScaler()
            X_norm[:, group] = scaler.fit_transform(X_norm[:, group])
        return X_norm

    def _pairwise_distance_from_vertical_parts(self, X: np.ndarray, features_groups: list[list[int]]):
        n_samples = X.shape[0]
        d_sq = np.zeros((n_samples, n_samples), dtype=float)

        def component_sqdist(group: list[int]) -> np.ndarray:
            Xg = X[:, group]
            diff = Xg[:, None, :] - Xg[None, :, :]
            return np.sum(diff * diff, axis=2)

        if len(features_groups) == 1 or self.n_jobs == 1:
            for group in features_groups:
                d_sq += component_sqdist(group)
        else:
            parts = Parallel(n_jobs=self.n_jobs)(delayed(component_sqdist)(group) for group in features_groups)
            for part in parts:
                if part is not None:
                    d_sq += part

        d_sq = np.maximum(d_sq, 0.0)
        return np.sqrt(d_sq)

    def _knn_indices(self, dist: np.ndarray, k: int):
        n_samples = dist.shape[0]
        if n_samples <= 1:
            return np.zeros((n_samples, 0), dtype=int)
        k = int(max(1, min(k, n_samples - 1)))
        dist_no_self = dist.copy()
        np.fill_diagonal(dist_no_self, np.inf)
        knn = np.argpartition(dist_no_self, kth=k - 1, axis=1)[:, :k]
        return knn

    def _compute_local_density(self, dist: np.ndarray, knn: np.ndarray):
        if knn.shape[1] == 0:
            return np.ones(dist.shape[0], dtype=float)
        row_idx = np.arange(dist.shape[0])[:, None]
        knn_dist = dist[row_idx, knn]
        eps = 1e-12
        return knn.shape[1] / (np.sum(knn_dist, axis=1) + eps)

    def _build_subclusters(self, dist: np.ndarray, knn: np.ndarray, rho: np.ndarray):
        n_samples = dist.shape[0]
        knn_mask = np.zeros((n_samples, n_samples), dtype=bool)
        for i in range(n_samples):
            knn_mask[i, knn[i]] = True

        mutual_mask = knn_mask & knn_mask.T
        representatives = np.full(n_samples, -1, dtype=int)

        for i in range(n_samples):
            candidates = np.flatnonzero(mutual_mask[i] & (rho > rho[i]))
            if candidates.size > 0:
                nearest = candidates[np.argmin(dist[i, candidates])]
                representatives[i] = int(nearest)

        order = np.argsort(-rho)
        subcluster_labels = np.full(n_samples, -1, dtype=int)
        next_label = 0

        for i in order:
            rep = representatives[i]
            if rep == -1:
                subcluster_labels[i] = next_label
                next_label += 1
            else:
                subcluster_labels[i] = subcluster_labels[rep]

        return subcluster_labels, representatives

    def _density_similarity(self, rho_i: np.ndarray, rho_j: np.ndarray):
        mu_i = float(np.mean(rho_i))
        mu_j = float(np.mean(rho_j))
        sd_i = float(np.std(rho_i))
        sd_j = float(np.std(rho_j))

        if mu_i == 0 and mu_j == 0:
            mean_term = 1.0
        else:
            mean_term = min(mu_i, mu_j) / max(mu_i, mu_j)

        if sd_i == 0 and sd_j == 0:
            std_term = 1.0
        else:
            std_term = 1.0 + (min(sd_i, sd_j) / max(sd_i, sd_j))

        return mean_term * std_term

    def _concentration(
        self,
        dist: np.ndarray,
        knn: np.ndarray,
        members_i: np.ndarray,
        members_j: np.ndarray,
    ):
        es_i = set(knn[members_i].ravel().tolist())
        es_j = set(knn[members_j].ravel().tolist())
        set_i = set(members_i.tolist())
        set_j = set(members_j.tolist())

        overlap = len(es_i.intersection(set_j)) + len(es_j.intersection(set_i))

        cross_dist = dist[np.ix_(members_i, members_j)].ravel()
        ni = len(members_i)
        nj = len(members_j)
        nij = int(np.ceil(min(0.1 * ni, 0.1 * nj)))
        nij = max(1, nij)
        d_min = float(np.mean(np.partition(cross_dist, nij - 1)[:nij]))

        return (1.0 + overlap) * (1.0 / (1.0 + d_min))

    def _subcluster_similarity_matrix(self, dist: np.ndarray, knn: np.ndarray, rho: np.ndarray, sub_labels: np.ndarray):
        unique_sub = np.unique(sub_labels)
        sub_to_idx = {int(s): idx for idx, s in enumerate(unique_sub)}
        idx_to_sub = {idx: int(s) for idx, s in enumerate(unique_sub)}

        n_sub = len(unique_sub)
        sim = np.eye(n_sub, dtype=float)

        members_by_sub = {}
        for s in unique_sub:
            members_by_sub[int(s)] = np.flatnonzero(sub_labels == s)

        pairs = [(a, b) for a in range(n_sub) for b in range(a + 1, n_sub)]

        def pair_similarity(pair: tuple[int, int]) -> tuple[int, int, float]:
            a, b = pair
            sa = idx_to_sub[a]
            sb = idx_to_sub[b]
            members_a = members_by_sub[sa]
            members_b = members_by_sub[sb]
            sim_rho = self._density_similarity(rho[members_a], rho[members_b])
            con = self._concentration(dist, knn, members_a, members_b)
            return a, b, sim_rho * con

        if self.n_jobs == 1 or len(pairs) == 0:
            results = [pair_similarity(pair) for pair in pairs]
        else:
            results = Parallel(n_jobs=self.n_jobs)(delayed(pair_similarity)(pair) for pair in pairs)

        for result in results:
            if result is None:
                continue
            a, b, value = result
            sim[a, b] = value
            sim[b, a] = value

        return sim, unique_sub, sub_to_idx

    def _merge_subclusters(self, sim_sub: np.ndarray, n_clusters: int):
        n_sub = sim_sub.shape[0]
        n_clusters = int(max(1, min(n_clusters, n_sub)))

        if n_sub == 1:
            return np.zeros(1, dtype=int)
        if n_clusters == n_sub:
            return np.arange(n_sub, dtype=int)

        seed = int(self.random_state.integers(0, int(1e9)))
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels=cast(Literal["kmeans", "discretize", "cluster_qr"], self.spectral_assign_labels),
            random_state=seed,
            n_init=self.spectral_n_init,
            n_jobs=self.n_jobs,
        )
        return sc.fit_predict(sim_sub)

    def _compute_cluster_centers(self, X: np.ndarray, labels: np.ndarray):
        unique_labels = np.unique(labels)
        centers = np.zeros((len(unique_labels), X.shape[1]), dtype=float)
        for i, lbl in enumerate(unique_labels):
            centers[i] = np.mean(X[labels == lbl], axis=0)
        return centers

    def _local_labels_from_global(self, X: np.ndarray, features_groups: list[list[int]], global_labels: np.ndarray):
        centers = self._compute_cluster_centers(X, global_labels)
        n_clusters = centers.shape[0]
        local_labels = []
        for group in features_groups:
            Xg = X[:, group]
            center_g = centers[:, group]
            d = np.linalg.norm(Xg[:, None, :] - center_g[None, :, :], axis=2)
            local_labels.append(np.argmin(d, axis=1))
        return local_labels

    def fit(  # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X, dtype=float)

        self._validate_inputs(X, features_groups)

        n_samples = X.shape[0]
        n_clusters = min(max(1, int(self.kmeans_n_clusters)), n_samples)

        X_norm = self._minmax_per_client(X, features_groups)
        dist = self._pairwise_distance_from_vertical_parts(X_norm, features_groups)

        k = int(max(1, min(self.knn_k, n_samples - 1))) if n_samples > 1 else 1
        knn = self._knn_indices(dist, k)
        rho = self._compute_local_density(dist, knn)

        sub_labels, representatives = self._build_subclusters(dist, knn, rho)
        sim_sub, unique_sub, sub_to_idx = self._subcluster_similarity_matrix(dist, knn, rho, sub_labels)

        merged_sub_labels = self._merge_subclusters(sim_sub, n_clusters)

        final_labels = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            final_labels[i] = merged_sub_labels[sub_to_idx[int(sub_labels[i])]]

        if self.use_server_labels:
            labels = final_labels
        else:
            labels = self._local_labels_from_global(X_norm, features_groups, final_labels)

        self.labels_ = labels
        self.global_labels_ = final_labels
        self.subcluster_labels_ = sub_labels
        self.representatives_ = representatives
        self.local_density_ = rho
        self.distance_matrix_ = dist
        self.subcluster_similarity_ = sim_sub
        self.n_clusters_ = len(np.unique(final_labels))
        self.n_subclusters_ = len(unique_sub)
        self.k_used_ = k

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
