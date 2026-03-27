import hashlib
from typing import Optional
from cohirf.experiment.clustering_experiment import ClusteringExperiment, calculate_scores
from vertcohirf.experiment.tested_models import models_dict
from ml_experiments.utils import profile_memory, profile_time
import numpy as np
import mlflow
import pandas as pd
from typing import Literal
from vertcohirf.experiment.vertibench.splitter import CorrelationSplitter, ImportanceSplitter
from pathlib import Path


def split_features_with_prob_and_cap(
    n_features, n_agents, p_overlap=0.2, max_overlap=0.3, rng_seed=None, sequential_split=True
):
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
    if sequential_split:
        base_splits = np.array_split(np.arange(n_features), n_agents)
    else:
        shuffled_features = rng.permutation(n_features)
        base_splits = np.array_split(shuffled_features, n_agents)
    agent_features = {i: set(split.tolist()) for i, split in enumerate(base_splits)}

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
        features_groups: Optional[list[list[int]]] = None,
        agent_i: Optional[int] = None,
        split_mode: Literal["sequential", "random", "importance", "correlation"] = "sequential",
        importance_splitter_weights: float | list[float] = 1.0,
        correlation_splitter_beta: float = 0.5,
        splitter_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Co-Clustering Experiment Initialization
        Args:
            n_agents : int, number of agents
            p_overlap : float, probability of duplicating a feature into another agent
            max_overlap : float in (0,1), maximum overlap fraction per agent
            features_groups : list of list of int, optional predefined feature groups for each agent
            agent_i : int, optional index of the agent to run (if None, run with all features)
            split_mode : str, mode for splitting features among agents ("sequential", "random", "importance", "correlation")
            importance_splitter_weights : float or list of float, weights for importance-based feature splitting (one or more float values)
            correlation_splitter_beta : float, beta parameter for correlation-based feature splitting
            splitter_dir : str, optional directory to save/load feature splits (for reproducibility and faster loading)
        """
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents
        self.p_overlap = p_overlap
        self.max_overlap = max_overlap
        self.features_groups = features_groups
        self.agent_i = agent_i
        self.split_mode = split_mode
        if isinstance(importance_splitter_weights, float):
            importance_splitter_weights = [importance_splitter_weights] * n_agents
        self.importance_splitter_weights = importance_splitter_weights
        self.correlation_splitter_beta = correlation_splitter_beta
        self.splitter_dir = splitter_dir

    @property
    def models_dict(self):
        return models_dict.copy()

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        if self.parser is None:
            raise ValueError("Parser is not initialized.")
        self.parser.add_argument("--n_agents", type=int, default=self.n_agents, help="Number of agents")
        self.parser.add_argument(
            "--p_overlap", type=float, default=self.p_overlap, help="Probability of feature overlap"
        )
        self.parser.add_argument("--max_overlap", type=float, default=self.max_overlap, help="Maximum feature overlap")
        # I am not sure if this is parsed correctly from command line, need to verify if we are using it
        self.parser.add_argument(
            "--features_groups", type=list, default=self.features_groups, help="Features groups for each agent"
        )
        self.parser.add_argument(
            "--agent_i",
            type=int,
            default=self.agent_i,
            help="If set, run only the model with partial data for this agent index",
        )
        self.parser.add_argument(
            "--split_mode", type=str, default=self.split_mode, help="Mode for splitting features among agents"
        )
        self.parser.add_argument(
            "--importance_splitter_weights",
            nargs="+",
            type=float,
            default=self.importance_splitter_weights,
            help="Weights for importance-based feature splitting (one or more float values)",
        )
        self.parser.add_argument(
            "--correlation_splitter_beta",
            type=float,
            default=self.correlation_splitter_beta,
            help="Beta parameter for correlation-based feature splitting",
        )
        self.parser.add_argument(
            "--splitter_dir",
            type=str,
            default=self.splitter_dir,
            help="Directory to save/load feature splits (for reproducibility and faster loading)",
        )

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.n_agents = args.n_agents
        self.p_overlap = args.p_overlap
        self.max_overlap = args.max_overlap
        self.features_groups = args.features_groups
        self.agent_i = args.agent_i
        self.split_mode = args.split_mode
        self.importance_splitter_weights = args.importance_splitter_weights
        self.correlation_splitter_beta = args.correlation_splitter_beta
        self.splitter_dir = args.splitter_dir
        return args

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params.update(
            {
                "n_agents": self.n_agents,
                "p_overlap": self.p_overlap,
                "max_overlap": self.max_overlap,
                "agent_i": self.agent_i,
                "split_mode": self.split_mode,
                "importance_splitter_weights": self.importance_splitter_weights,
                "correlation_splitter_beta": self.correlation_splitter_beta,
                "splitter_dir": self.splitter_dir,
            }
        )
        return unique_params

    def _get_extra_params(self):
        extra_params = super()._get_extra_params()
        extra_params.update(
            {
                "features_groups": self.features_groups,
            }
        )
        return extra_params

    def _after_load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        n_agents = unique_params["n_agents"]
        p_overlap = unique_params["p_overlap"]
        max_overlap = unique_params["max_overlap"]
        split_mode = unique_params["split_mode"]
        importance_splitter_weights = unique_params["importance_splitter_weights"]
        correlation_splitter_beta = unique_params["correlation_splitter_beta"]
        splitter_dir = unique_params["splitter_dir"]
        features_groups = extra_params["features_groups"]
        splitter = None
        X = kwargs["load_data_return"]["X"]
        if "seed_dataset" in combination:
            seed_dataset = combination["seed_dataset"]
        elif "seed_dataset_order" in combination:
            seed_dataset = combination["seed_dataset_order"]
        else:
            raise ValueError("No seed for dataset found in combination")
        if features_groups is None:
            # this is a ugly way to just compute faster the splits for the case of real datasets by storing the results
            if "dataset_id" in combination:
                dataset_id = combination["dataset_id"]
                split_hash = f"dataset-{dataset_id}_seed-{seed_dataset}_nagents-{n_agents}_poverlap-{p_overlap}_maxoverlap-{max_overlap}_splitmode-{split_mode}_importanceweights-{'-'.join(map(str, importance_splitter_weights))}_correlationbeta-{correlation_splitter_beta}"
                split_hash = hashlib.sha256(split_hash.encode("utf-8")).hexdigest()[:10]  # shorten the name
                if splitter_dir is not None:
                    # Check if split cache exists before recomputing.
                    # Prefer NPZ to preserve NumPy arrays (dtype/shape) and load faster than JSON.
                    split_path = Path(splitter_dir) / f"{split_hash}.npz"
                    if split_path.exists():
                        with np.load(split_path, allow_pickle=False) as data:
                            # np.load returns NumPy arrays; normalize to plain Python lists for stable downstream types.
                            keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
                            features_groups = [data[key].astype(np.int64, copy=False).tolist() for key in keys]
                    else:
                        features_groups = None
            else: 
                features_groups = None
                split_hash = None
            if features_groups is None:
                if split_mode in ["sequential", "random"]:
                    sequential_split = split_mode == "sequential"
                    features_groups = split_features_with_prob_and_cap(
                        X.shape[1],
                        n_agents=n_agents,
                        p_overlap=p_overlap,
                        max_overlap=max_overlap,
                        rng_seed=seed_dataset,
                        sequential_split=sequential_split,
                    )
                elif split_mode in ["importance", "correlation"]:
                    if split_mode == "importance":
                        splitter = ImportanceSplitter(
                            num_parties=n_agents,
                            weights=importance_splitter_weights, 
                            seed=seed_dataset,
                        )
                    else:  # split_mode == "correlation":
                        splitter = CorrelationSplitter(num_parties=n_agents, seed=seed_dataset)
                        if isinstance(X, pd.DataFrame):
                            X_values = X.values # CorrelationSplitter expects a numpy array
                        else:
                            X_values = X
                        splitter.fit(X_values)
                    features_groups = splitter.split_indices(X, allow_empty_party=False, beta=correlation_splitter_beta)
                else:
                    raise ValueError(f"Unknown split mode: {split_mode}")
                if splitter_dir is not None and split_hash is not None:
                    split_path = Path(splitter_dir) / f"{split_hash}.npz"
                    split_path.parent.mkdir(parents=True, exist_ok=True)
                    # Save variable-length feature groups as separate arrays in a compressed NPZ archive.
                    np.savez_compressed(split_path, *[np.asarray(group, dtype=np.int64) for group in features_groups])
        else:
            # features groups was provided as an input, we log it to mlflow if available
            if mlflow_run_id is not None:
                mlflow.log_params({"features_groups_": features_groups}, run_id=mlflow_run_id)
        if features_groups is not None:
            features_groups = [list(map(int, group)) for group in features_groups]
        ret = super()._after_load_data(combination, unique_params, extra_params, mlflow_run_id, **kwargs)
        ret["features_groups"] = features_groups
        ret["splitter"] = splitter if split_mode in ["importance", "correlation"] else None
        return ret

    @profile_time(enable_based_on_attribute="profile_time")
    @profile_memory(enable_based_on_attribute="profile_memory")
    def _fit_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        agent_i = unique_params["agent_i"]
        model = kwargs["load_model_return"]["model"]
        X = kwargs["load_data_return"]["X"]
        features_groups = kwargs["after_load_data_return"]["features_groups"]
        if agent_i is None:
            y_pred = model.fit_predict(X, features_groups=features_groups)
        else:
            if isinstance(X, pd.DataFrame):
                y_pred = model.fit_predict(X.iloc[:, features_groups[agent_i]])
            else:
                y_pred = model.fit_predict(X[:, features_groups[agent_i]])
        return {"y_pred": y_pred}

    @profile_time(enable_based_on_attribute="profile_time")
    @profile_memory(enable_based_on_attribute="profile_memory")
    def _evaluate_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        agent_i = unique_params["agent_i"]
        scores = kwargs["get_metrics_return"]
        X = kwargs["load_data_return"]["X"]
        y_true = kwargs["load_data_return"]["y"]
        # if multiple y_true, we suppose y[0] is the global label and y[1:], if any, are local labels
        if isinstance(y_true, (tuple, list)):
            if agent_i is None:
                y_true = y_true[0]
            else:
                y_true = y_true[agent_i + 1]
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
            results_i = calculate_scores(
                calculate_metrics_even_if_too_many_clusters, min_n_clusters, X, y_true, y_pred_i, scores
            )
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
