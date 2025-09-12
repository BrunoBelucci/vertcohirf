from typing import Optional
from cohirf.experiment.clustering_experiment import ClusteringExperiment, calculate_scores
from cohirf.experiment.open_ml_clustering_experiment import OpenmlClusteringExperiment
from cocohirf.experiment.tested_models import models_dict
from ml_experiments.utils import profile_memory, profile_time
import numpy as np


def split_features_with_prob_and_cap(n_features, n_agents, p_overlap=0.2, max_overlap=0.3, rng_seed=None):
    """
    Partition features across agents, then with probability p_overlap
    copy features to other agents, but stop when an agent reaches max_overlap.

    Args:
        X : np.ndarray, shape (n_samples, n_features)
        n_agents : int, number of agents
        p_overlap : float, probability of duplicating a feature into another agent
        max_overlap : float in (0,1), maximum overlap fraction per agent
        rng_seed : int, optional random seed
    Returns:
        list of np.ndarray, each of shape (n_samples, q_i)
    """
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    # Step 1: Base partition
    base_splits = np.array_split(np.arange(n_features), n_agents)
    agent_features = {i: set(split) for i, split in enumerate(base_splits)}

    # Step 2: Track overlap limits
    max_allowed = {i: int(len(split) * max_overlap) for i, split in enumerate(base_splits)}
    current_overlap = {i: 0 for i in range(n_agents)}

    # Step 3: Per-feature probabilistic overlap
    for feat in range(n_features):
        src_agent = [i for i, s in enumerate(base_splits) if feat in s][0]

        for dst_agent in range(n_agents):
            if dst_agent == src_agent:
                continue
            if current_overlap[dst_agent] < max_allowed[dst_agent]:
                if rng.random() < p_overlap:
                    agent_features[dst_agent].add(feat)
                    current_overlap[dst_agent] += 1

    # Step 4: Build final list of arrays
    return [sorted(agent_features[i]) for i in range(n_agents)]


class CoClusteringExperiment(ClusteringExperiment):

    def __init__(
        self,
        *args,
        n_agents: int = 2,
        p_overlap: float = 0.2,
        max_overlap: float = 0.3,
        **kwargs
    ):
        """
        Co-Clustering Experiment Initialization
        Args:
            n_agents : int, number of agents
            p_overlap : float, probability of duplicating a feature into another agent
            max_overlap : float in (0,1), maximum overlap fraction per agent
        """
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents
        self.p_overlap = p_overlap
        self.max_overlap = max_overlap

    @property
    def models_dict(self):
        return models_dict.copy()

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        if self.parser is None:
            raise ValueError("Parser is not initialized.")
        self.parser.add_argument('--n_agents', type=int, default=self.n_agents, help='Number of agents')
        self.parser.add_argument('--p_overlap', type=float, default=self.p_overlap, help='Probability of feature overlap')
        self.parser.add_argument('--max_overlap', type=float, default=self.max_overlap, help='Maximum feature overlap')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.n_agents = args.n_agents
        self.p_overlap = args.p_overlap
        self.max_overlap = args.max_overlap
        return args

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params.update({
            "n_agents": self.n_agents,
            "p_overlap": self.p_overlap,
            "max_overlap": self.max_overlap,
        })
        return unique_params

    @profile_time(enable_based_on_attribute="profile_time")
    @profile_memory(enable_based_on_attribute="profile_memory")
    def _fit_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        n_agents = unique_params["n_agents"]
        p_overlap = unique_params["p_overlap"]
        max_overlap = unique_params["max_overlap"]
        if "seed_dataset" in combination:
            seed_dataset = combination["seed_dataset"]
        elif "seed_dataset_order" in combination:
            seed_dataset = combination["seed_dataset_order"]
        else:
            raise ValueError("No seed for dataset found in combination")
        model = kwargs["load_model_return"]["model"]
        X = kwargs["load_data_return"]["X"]
        features_groups = split_features_with_prob_and_cap(
            X.shape[1],
            n_agents=n_agents,
            p_overlap=p_overlap,
            max_overlap=max_overlap,
            rng_seed=seed_dataset,
        )
        y_pred = model.fit_predict(X, features_groups=features_groups)
        return {"y_pred": y_pred}

    @profile_time(enable_based_on_attribute="profile_time")
    @profile_memory(enable_based_on_attribute="profile_memory")
    def _evaluate_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        scores = kwargs["get_metrics_return"]
        X = kwargs["load_data_return"]["X"]
        y_true = kwargs["load_data_return"]["y"]
        y_pred = kwargs["fit_model_return"]["y_pred"]
        calculate_metrics_even_if_too_many_clusters = unique_params["calculate_metrics_even_if_too_many_clusters"]
        results = {}
        if not isinstance(y_pred, tuple) and not isinstance(y_pred, list):
            # if there is only one partition, make it a list for easier handling and generalization
            # if there are multiple partitions (e.g. different agents), keep it as is
            y_pred = [y_pred]

        # get number of clusters
        n_clusters = []
        for i, y_pred_i in enumerate(y_pred):
            results[f"n_clusters_{i}_"] = len(np.unique(y_pred_i))
            n_clusters.append(len(np.unique(y_pred_i)))
        min_n_clusters = min(n_clusters)
        results["min_n_clusters"] = min_n_clusters

        # get results for each agent
        results_list = []
        for i, y_pred_i in enumerate(y_pred):
            results_i = calculate_scores(calculate_metrics_even_if_too_many_clusters, min_n_clusters, X, y_true, y_pred_i, scores)
            results_list.append(results_i)
        # aggregate results
        for metric in results_list[0].keys():
            results[f"{metric}_mean"] = np.mean([r[metric] for r in results_list])
            results[f"{metric}_std"] = np.std([r[metric] for r in results_list])
        # add individual results
        for i, results_i in enumerate(results_list):
            for metric, value in results_i.items():
                results[f"{metric}_{i}"] = value
        return results
