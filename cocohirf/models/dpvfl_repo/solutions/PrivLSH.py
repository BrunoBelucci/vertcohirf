import numpy as np
# import copy
from sklearn.cluster import KMeans
import itertools
import logging
import copy

from .VBase import VBase
from ..util.save_results import save_result_to_json
from ..util.eval_centers import eval_centers
from ..util.volh import volh_perturb, volh_membership, rr_membership, rr_perturb
from ..util.load_config import generate_local_config
from ..util.fmsketch import intersection_ca
from ..util.postprocess import norm_sub

from .lsh_clustering.clustering_algorithm import get_private_coreset, private_lsh_clustering, ClusteringResult
from .lsh_clustering.clustering_params import DifferentialPrivacyParam, Data

'''
This is an implementation of the paper Hu Ding et al. "K-Means Clustering with Distributed Dimensions"
The paper introduces a non-private VFL solution for K-means clustering problem
'''


class PrivLSH(VBase):
    def __init__(self, config, tag, save_result=False, **kwargs):
        super().__init__(config, tag, )
        self.config = config
        self.tag = tag
        self.k = config['k']
        assert 'eps' in config
        self.eps = config['eps']
        if self.eps == 0:
            self.eps = np.inf

        self.cluster_centers_ = None
        self.save_result = save_result
        self._random_state = config["random_state"] if "random_state" in config else None

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

    def fit(self, data, run_clean: bool = True):
        if isinstance(data, list):
            assert len(data) == 1
            data = data[0]
        # logging.info(f"locally use LSH-clustering to fit with data shape:{data.shape}, epsilon={self.eps}")
        # the google's version user gaussian noise for average, here we need to use the Laplace noise
        data_obj = Data(data, radius=1)
        result: ClusteringResult = private_lsh_clustering(self.k, data_obj,
                                                          privacy_param=DifferentialPrivacyParam(self.eps, delta=1e-6))
        final_loss = eval_centers(data, result.centers)
        # logging.info(f"local kmeans loss: {final_loss}")
        if self.save_result:
            self.save_results(final_loss, result.centers)
        self.cluster_centers_ = result.centers
        return result.centers

    def save_results(self, loss, centers):
        results = {
            "config": self.config,
            "losses": {"private_final_loss": loss},
            "final_centers": centers,
        }
        save_result_to_json(results, self.tag, experiment=self.config['dataset'] + "-central-clustering")
