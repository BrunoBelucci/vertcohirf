from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import pandas as pd
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


class VFCkM(ClusterMixin, BaseEstimator):

    def __init__(
        self,
        random_state=None,
        n_jobs: int = 1,
        # kmeans parameters
        kmeans_n_clusters: int = 8,
        kmeans_tol: float = 1e-4,
        # VFCkM outer-loop parameters
        max_global_iter: int = 100,
        global_tol: float = 1e-10,
        center_matching: str = "paper_greedy",
    ):
        self._random_state = random_state
        self.n_jobs = n_jobs
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_tol = kmeans_tol
        self.max_global_iter = max_global_iter
        self.global_tol = global_tol
        self.center_matching = center_matching

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
        if len(features_groups) == 0:
            raise ValueError("features_groups must contain at least one client feature group.")
        n_features = X.shape[1]
        for i, group in enumerate(features_groups):
            if len(group) == 0:
                raise ValueError(f"features_groups[{i}] cannot be empty.")
            invalid = [f for f in group if f < 0 or f >= n_features]
            if invalid:
                raise ValueError(f"features_groups[{i}] contains invalid feature indices: {invalid}")

    def _run_local_one_step(
        self,
        X: np.ndarray,
        group: list[int],
        n_clusters: int,
        sample_weight: np.ndarray,
        init_centers: np.ndarray | None,
        seed: int,
    ):
        X_group = X[:, group]
        if init_centers is None:
            init = "random"
        else:
            init = init_centers
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=1,
            max_iter=1,
            tol=self.kmeans_tol,
            random_state=seed,
        )
        labels = kmeans.fit_predict(X_group, sample_weight=sample_weight)
        centers = kmeans.cluster_centers_
        sq_dist = np.sum((X_group - centers[labels]) ** 2, axis=1)
        weights = np.bincount(labels, weights=sample_weight, minlength=n_clusters)
        error = float(np.sum(sample_weight * sq_dist))
        return labels, centers, weights, error

    def _initial_local_step(
        self, X: np.ndarray, features_groups: list[list[int]], n_clusters: int, sample_weight: np.ndarray
    ):
        seeds = self.random_state.integers(0, int(1e9), size=len(features_groups))
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_local_one_step)(
                X=X,
                group=group,
                n_clusters=n_clusters,
                sample_weight=sample_weight,
                init_centers=None,
                seed=int(seed),
            )
            for group, seed in zip(features_groups, seeds)
        )

    def _iter_local_step(
        self,
        X: np.ndarray,
        features_groups: list[list[int]],
        n_clusters: int,
        sample_weight: np.ndarray,
        global_centers: np.ndarray,
    ):
        seeds = self.random_state.integers(0, int(1e9), size=len(features_groups))
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_local_one_step)(
                X=X,
                group=group,
                n_clusters=n_clusters,
                sample_weight=sample_weight,
                init_centers=global_centers[:, group],
                seed=int(seed),
            )
            for group, seed in zip(features_groups, seeds)
        )

    def _align_to_reference(
        self,
        reference_centers: np.ndarray,
        reference_group: list[int],
        centers: np.ndarray,
        group: list[int],
    ):
        shared = sorted(set(reference_group).intersection(group))
        if len(shared) == 0:
            return np.arange(reference_centers.shape[0])

        ref_idx = [reference_group.index(f) for f in shared]
        grp_idx = [group.index(f) for f in shared]

        ref_proj = reference_centers[:, ref_idx]
        grp_proj = centers[:, grp_idx]
        cost = np.linalg.norm(ref_proj[:, None, :] - grp_proj[None, :, :], axis=2)

        if self.center_matching == "hungarian":
            row_ind, col_ind = linear_sum_assignment(cost)
            order = np.zeros(reference_centers.shape[0], dtype=int)
            order[row_ind] = col_ind
            return order

        if self.center_matching != "paper_greedy":
            raise ValueError("center_matching must be either 'paper_greedy' or 'hungarian'.")

        # Paper-style pairing: repeatedly pick the minimum difference pair, then remove that row/column.
        n_rows, n_cols = cost.shape
        if n_rows != n_cols:
            raise ValueError("VFCkM requires equal numbers of local and reference centers for matching.")
        order = np.full(n_rows, -1, dtype=int)
        remaining_rows = set(range(n_rows))
        remaining_cols = set(range(n_cols))
        while remaining_rows:
            best_pair = None
            best_cost = np.inf
            for r in remaining_rows:
                for c in remaining_cols:
                    value = cost[r, c]
                    if value < best_cost:
                        best_cost = value
                        best_pair = (r, c)
            if best_pair is None:
                raise RuntimeError("Failed to find a valid center pair during greedy matching.")
            r, c = best_pair
            order[r] = c
            remaining_rows.remove(r)
            remaining_cols.remove(c)
        return order

    def _merge_global_centers(self, local_results, features_groups: list[list[int]], n_features: int):
        labels_i, centers_i, weights_i, errors_i = zip(*local_results)

        reference_client = int(np.argmax([len(g) for g in features_groups]))
        reference_centers = centers_i[reference_client]
        reference_group = features_groups[reference_client]
        n_clusters = reference_centers.shape[0]

        aligned_centers = []
        aligned_weights = []
        for i, group in enumerate(features_groups):
            if i == reference_client:
                order = np.arange(n_clusters)
            else:
                order = self._align_to_reference(reference_centers, reference_group, centers_i[i], group)
            aligned_centers.append(centers_i[i][order])
            aligned_weights.append(weights_i[i][order])

        aligned_centers = np.asarray(aligned_centers)
        aligned_weights = np.asarray(aligned_weights)
        global_centers = np.zeros((n_clusters, n_features), dtype=float)

        feature_to_clients = {}
        for client_id, group in enumerate(features_groups):
            for local_idx, feature_idx in enumerate(group):
                feature_to_clients.setdefault(feature_idx, []).append((client_id, local_idx))

        # Follow the paper merge rule on shared dimensions: keep the center from the highest-weight local cluster.
        for k in range(n_clusters):
            for feature_idx, providers in feature_to_clients.items():
                candidate_weights = np.array([aligned_weights[cid, k] for cid, _ in providers])
                winner = int(np.argmax(candidate_weights))
                winner_client, winner_local_idx = providers[winner]
                global_centers[k, feature_idx] = aligned_centers[winner_client, k, winner_local_idx]

        total_error = float(np.sum(errors_i))
        # final_labels = np.asarray(labels_i[reference_client])
        return global_centers, total_error, labels_i

    def fit(  # type: ignore
        self,
        X: pd.DataFrame | np.ndarray,
        features_groups: list[list[int]],
        y=None,
        sample_weight=None,
    ):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X)
        self._validate_inputs(X, features_groups)

        n_samples, n_features = X.shape
        n_clusters = min(self.kmeans_n_clusters, n_samples)
        sample_weight = (
            np.ones(n_samples, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )

        local_results = self._initial_local_step(X, features_groups, n_clusters, sample_weight)
        global_centers, te, labels = self._merge_global_centers(local_results, features_groups, n_features)

        pte = np.inf
        n_outer_iter = 0
        while te < (pte - self.global_tol) and n_outer_iter < self.max_global_iter:
            pte = te
            local_results = self._iter_local_step(X, features_groups, n_clusters, sample_weight, global_centers)
            global_centers, te, labels = self._merge_global_centers(local_results, features_groups, n_features)
            n_outer_iter += 1

        self.cluster_centers_ = global_centers
        self.labels_ = labels
        self.total_error_ = te
        self.previous_total_error_ = pte
        self.n_iter_ = n_outer_iter
        self.n_clusters_ = n_clusters
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
