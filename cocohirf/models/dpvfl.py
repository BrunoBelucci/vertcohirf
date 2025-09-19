from sklearn.base import BaseEstimator, ClusterMixin
import pandas as pd
import numpy as np
from .dpvfl_repo.solutions.V2way import V2way
from .dpvfl_repo.solutions.VPrivClustering import VPrivClustering
from typing import Literal

# The code comes from https://anonymous.4open.science/r/public_vflclustering-63CD mentioned in "Differentially Private Vertical Federated Clustering"


class DPVFL(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_clusters: int = 8,  # k
        m_div: int = 10,
        eps: float = 0.5,
        intersection_method: str = 'fmsketch',
        random_state: int | None = None,
        local_solver: str = 'lsh_clustering',
        mode: Literal['v2way', 'vpc'] = 'v2way'
    ):
        self.n_clusters = n_clusters
        self.m_div = m_div
        self.eps = eps
        self.intersection_method = intersection_method
        self.random_state = random_state
        self.local_solver = local_solver
        self.mode = mode

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

        n_agents = len(features_groups)  # T
        n_samples = X.shape[0]  # n
        # seems like d is not used
        config = {
            'k': self.n_clusters,
            'm': n_samples // self.m_div,
            'T': n_agents,
            'eps': self.eps,
            'intersection_method': self.intersection_method,
            'random_state': self.random_state,
            'local_solver': self.local_solver,
            'n': n_samples,
            'd': None # it is not used inside the method even though it is accessed in generate_local_config
        }
        if self.mode == 'v2way':
            model_cls = V2way
        elif self.mode == 'vpc':
            model_cls = VPrivClustering
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        model = model_cls(config=config, tag=self.mode)
        data = [X[:, features] for features in features_groups]
        # I am not sure if we should use final_solver or clearn_final_solver
        final_solver = model.fit(data, run_clean=True)
        clusters_centers = final_solver.cluster_centers_
        labels = []
        for i in range(n_agents):
            X_group = X[:, features_groups[i]]
            dist = [
                np.linalg.norm(X_group - clusters_centers[j][features_groups[i]], axis=1)
                for j in range(self.n_clusters)
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
