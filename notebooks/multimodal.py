# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: cocohirf
#     language: python
#     name: python3
# ---

# %%
from cocohirf.experiment.coclustering_experiment import CoClusteringExperiment
from cocohirf.experiment.hpo_vecohirf_experiment import HPOVeCoHiRFExperiment
from cohirf.experiment.clustering_experiment import ClusteringExperiment
from cohirf.experiment.hpo_clustering_experiment import HPOClusteringExperiment
from ml_experiments.hpo_experiment import HPOExperiment
from cohirf.experiment.open_ml_clustering_experiment import preprocess
from cocohirf.experiment.tested_models import models_dict
import pandas as pd
from typing import Optional
from pathlib import Path
import numpy as np
from IPython.display import clear_output


# %%
class DataClusteringExperiment(ClusteringExperiment):
    def __init__(self, *args, X_data, y_data, cat_features, seed_dataset_order, standardize, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_data = X_data
        self.y_data = y_data
        self.cat_features = cat_features
        self.seed_dataset_order = seed_dataset_order
        self.standardize = standardize

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params["X_data"] = self.X_data
        unique_params["y_data"] = self.y_data
        unique_params["cat_features"] = self.cat_features
        unique_params["standardize"] = self.standardize
        return unique_params

    def _get_combinations_names(self) -> list[str]:
        combination_names = super()._get_combinations_names()
        combination_names.extend(["seed_dataset_order"])
        return combination_names

    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        X_data = unique_params["X_data"]
        y_data = unique_params["y_data"]
        cat_features = unique_params["cat_features"]
        standardize = unique_params["standardize"]
        seed_dataset_order = combination["seed_dataset_order"]
        if cat_features is not None:
            # cat features is a string with the format 'feature1,feature2,feature3' transform it to list of integers
            cat_features = [int(feature) for feature in cat_features.split(',')]
        else:
            cat_features = []
        cat_features_names = X_data.columns[cat_features].tolist()
        cont_features_names = [feature for feature in X_data.columns if feature not in cat_features_names]
        n_classes = len(y_data.unique())
        X, y = preprocess(X_data, y_data, cat_features_names, cont_features_names, standardize, seed_dataset_order)
        return {
            'X': X,
            'y': y,
            'cat_features_names': cat_features_names,
            'cont_features_names': cont_features_names,
            'n_classes': n_classes,
        }

class HPODataClusteringExperiment(HPOClusteringExperiment, DataClusteringExperiment):

    def _load_simple_experiment(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        experiment = DataClusteringExperiment(
            X_data=unique_params["X_data"],
            y_data=unique_params["y_data"],
            cat_features=unique_params["cat_features"],
            seed_dataset_order=unique_params["seed_dataset_order"],
            standardize=unique_params["standardize"],
            # experiment parameters
            experiment_name=self.experiment_name,
            log_dir=self.log_dir,
            log_file_name=self.log_file_name,
            work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir,
            clean_work_dir=self.clean_work_dir,
            clean_data_dir=False,
            raise_on_error=self.raise_on_error,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            check_if_exists=self.check_if_exists,
            profile_memory=self.profile_memory,
            profile_time=self.profile_time,
            verbose=0,
        )
        return experiment


class DataCoClusteringExperiment(DataClusteringExperiment, CoClusteringExperiment):

    @property
    def models_dict(self):
        return models_dict.copy()

class HPODataCoClusteringExperiment(HPOClusteringExperiment, DataCoClusteringExperiment):

    def _load_simple_experiment(self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs):
        experiment = DataCoClusteringExperiment(
            X_data=unique_params["X_data"],
            y_data=unique_params["y_data"],
            cat_features=unique_params["cat_features"],
            seed_dataset_order=combination["seed_dataset_order"],
            standardize=unique_params["standardize"],
            # experiment parameters
            experiment_name=self.experiment_name,
            log_dir=self.log_dir,
            log_file_name=self.log_file_name,
            work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir,
            clean_work_dir=self.clean_work_dir,
            clean_data_dir=False,
            raise_on_error=self.raise_on_error,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            check_if_exists=self.check_if_exists,
            profile_memory=self.profile_memory,
            profile_time=self.profile_time,
            verbose=0,
        )
        return experiment

class HPODataVeCoHiRFExperiment(HPOVeCoHiRFExperiment, DataCoClusteringExperiment):

    @property
    def stage_1_experiment(self):
        return DataClusteringExperiment

    @property
    def stage_2_experiment(self):
        return DataCoClusteringExperiment

    def get_dataset_parameters(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        X_data = unique_params["X_data"]
        y_data = unique_params["y_data"]
        cat_features = unique_params["cat_features"]
        seed_dataset_order = combination["seed_dataset_order"]
        standardize = unique_params["standardize"]
        return {
            "X_data": X_data,
            "y_data": y_data,
            "cat_features": cat_features,
            "seed_dataset_order": seed_dataset_order,
            "standardize": standardize,
        }


# %%
data_dir = Path("/home/belucci/code/cohirf/data/wine")
X_data = pd.read_csv(data_dir / "X.csv", index_col=0)
y_data = pd.read_csv(data_dir / "y.csv", index_col=0)
description_group = [i for i, col in enumerate(X_data.columns) if "description" in col]
other_group = [i for i in range(X_data.shape[1]) if i not in description_group]
features_groups = [description_group, other_group]

# %%
results_dir = Path("/home/belucci/code/cocohirf/results") / "modal"
results_dir.mkdir(parents=True, exist_ok=True)
mlflow_tracking_uri = f"sqlite:///{results_dir}/mlflow.db"
experiment_params = dict(
    mlflow_tracking_uri=mlflow_tracking_uri,
    check_if_exists=False,
)

# %% [markdown]
# # KMeans total

# %%
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand"
for seed in seeds:
    selected_indexes = X_data.sample(n=10_000, random_state=seed).index
    X = X_data.loc[selected_indexes].reset_index(drop=True)
    y = y_data.loc[selected_indexes].reset_index(drop=True)
    experiment = HPODataClusteringExperiment(
        # hpo
        n_trials=30,
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        # model
        model="KMeans",
        seed_model=seed,
        # dataset
        X_data=X,
        y_data=y.iloc[:, 0],
        cat_features=None,
        seed_dataset_order=seed,
        standardize=True,
        # experiment parameters
        raise_on_error=True,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand_mean"
for seed in seeds:
    selected_indexes = X_data.sample(n=10_000, random_state=seed).index
    X = X_data.loc[selected_indexes].reset_index(drop=True)
    y = y_data.loc[selected_indexes].reset_index(drop=True)
    experiment = HPODataCoClusteringExperiment(
        # hpo
        n_trials=30,
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        # model
        model="KMeans",
        seed_model=seed,
        # dataset
        X_data=X,
        y_data=y.iloc[:, 0],
        cat_features=None,
        seed_dataset_order=seed,
        standardize=True,
        features_groups=features_groups,
        agent_i=0,
        # experiment parameters
        raise_on_error=True,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand_mean"
for seed in seeds:
    selected_indexes = X_data.sample(n=10_000, random_state=seed).index
    X = X_data.loc[selected_indexes].reset_index(drop=True)
    y = y_data.loc[selected_indexes].reset_index(drop=True)
    # drop 0 std features and also get rid of them in features_groups
    stds = X.std()
    zero_std_features = stds[stds == 0].index.tolist()
    X = X.drop(columns=zero_std_features)
    description_group = [i for i, col in enumerate(X.columns) if "description" in col]
    other_group = [i for i in range(X.shape[1]) if i not in description_group]
    features_groups = [description_group, other_group]
    experiment = HPODataCoClusteringExperiment(
        # hpo
        n_trials=30,
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        # model
        model="KMeans",
        seed_model=seed,
        # dataset
        X_data=X,
        y_data=y.iloc[:, 0],
        cat_features=None,
        seed_dataset_order=seed,
        standardize=True,
        features_groups=features_groups,
        agent_i=1,
        # experiment parameters
        raise_on_error=True,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand_mean"
for seed in seeds:
    selected_indexes = X_data.sample(n=10_000, random_state=seed).index
    X = X_data.loc[selected_indexes].reset_index(drop=True)
    y = y_data.loc[selected_indexes].reset_index(drop=True)
    # drop 0 std features and also get rid of them in features_groups
    stds = X.std()
    zero_std_features = stds[stds == 0].index.tolist()
    X = X.drop(columns=zero_std_features)
    description_group = [i for i, col in enumerate(X.columns) if "description" in col]
    other_group = [i for i in range(X.shape[1]) if i not in description_group]
    features_groups = [description_group, other_group]
    experiment = HPODataVeCoHiRFExperiment(
        # hpo
        n_trials_1=30,
        n_trials_2=30,
        hpo_seed=seed,
        hpo_metric_1=hpo_metric[: -len("_mean")],
        hpo_metric_2=hpo_metric,
        direction_1="maximize",
        direction_2="maximize",
        # model
        model_alias="VeCoHiRF",
        seed_model=seed,
        # dataset
        X_data=X,
        y_data=y.iloc[:, 0],
        cat_features=None,
        seed_dataset_order=seed,
        standardize=True,
        features_groups=features_groups,
        # agent_i=1,
        # experiment parameters
        raise_on_error=True,
        verbose=1,
        **experiment_params,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
metrics
