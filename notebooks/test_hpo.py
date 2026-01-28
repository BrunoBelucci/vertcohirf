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
from cocohirf.experiment.hpo_open_ml_vecohirf_experiment import HPOOpenmlVeCoHiRFExperiment
from cocohirf.experiment.hpo_open_ml_coclustering_experiment import HPOOpenmlCoClusteringExperiment
from cohirf.models.cohirf import BaseCoHiRF
from cohirf.models.vecohirf import VeCoHiRF
from cohirf.experiment.hpo_open_ml_clustering_experiment import HPOOpenmlClusteringExperiment
from cocohirf.models.distributed_kmeans import DistributedKMeans
from cocohirf.models.coreset_kmeans import CoresetKMeans
from sklearn.cluster import KMeans
import optuna
import numpy as np
from IPython.display import clear_output

# %% [markdown]
# # KMeans

# %%
model = KMeans
model_params = dict()
search_space = dict(
	n_clusters = optuna.distributions.IntDistribution(2, 30),
)
default_values = [
	dict(n_clusters=8),
]
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand"
for seed in seeds:
    experiment = HPOOpenmlClusteringExperiment(
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        search_space=search_space,
        default_values=default_values,
        model=model,
        model_params=model_params,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### experiment parameters
        experiment_name="kmeans",
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %% [markdown]
# # CoHiRF

# %%
model = BaseCoHiRF
model_params = dict()
search_space = dict(
    n_features = optuna.distributions.FloatDistribution(0.1, 1.0),
    repetitions = optuna.distributions.IntDistribution(2, 10),
    base_model_kwargs=dict(
        n_clusters=optuna.distributions.IntDistribution(2, 30),
    )
)
default_values = [
    dict(n_features=0.6, repetitions=3, base_model_kwargs=dict(n_clusters=8)),
]
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand"
for seed in seeds:
    experiment = HPOOpenmlClusteringExperiment(
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        search_space=search_space,
        default_values=default_values,
        model=model,
        model_params=model_params,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### experiment parameters
        experiment_name="kmeans",
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
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
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %% [markdown]
# # DistributedKMeans

# %%
model = DistributedKMeans
model_params = dict()
search_space = dict(
	kmeans_n_clusters=optuna.distributions.IntDistribution(2, 30),
)
default_values = [
    dict(kmeans_n_clusters=8),
]
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand_mean"
for seed in seeds:
    experiment = HPOOpenmlCoClusteringExperiment(
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        search_space=search_space,
        default_values=default_values,
        model=model,
        model_params=model_params,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### experiment parameters
        experiment_name=model.__name__,
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
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
model = DistributedKMeans
model_params = dict(use_server_labels=False)
search_space = dict(
    kmeans_n_clusters=optuna.distributions.IntDistribution(2, 30),
)
default_values = [
    dict(kmeans_n_clusters=8),
]
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand_mean"
for seed in seeds:
    experiment = HPOOpenmlCoClusteringExperiment(
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        search_space=search_space,
        default_values=default_values,
        model=model,
        model_params=model_params,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### experiment parameters
        experiment_name=model.__name__,
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %% [markdown]
# # CoresetKMeans

# %%
model = CoresetKMeans
model_params = dict()
search_space = dict(
    kmeans_n_clusters=optuna.distributions.IntDistribution(2, 30),
)
default_values = [
    dict(kmeans_n_clusters=8),
]
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand_mean"
for seed in seeds:
    experiment = HPOOpenmlCoClusteringExperiment(
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        search_space=search_space,
        default_values=default_values,
        model=model,
        model_params=model_params,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### experiment parameters
        experiment_name=model.__name__,
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %% [markdown]
# # VeCoHiRF
#

# %% [markdown]
# ## Naive

# %%
model = VeCoHiRF
model_params = dict(
    cohirf_kwargs=[
        dict(max_iter=1), dict(max_iter=1),
    ]
)
search_space = dict(
    cohirf_kwargs=[
        dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
			repetitions=optuna.distributions.IntDistribution(2, 10),
            base_model_kwargs=dict(
				n_clusters=optuna.distributions.IntDistribution(2, 30),
			)
		),
        dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
			repetitions=optuna.distributions.IntDistribution(2, 10),
            base_model_kwargs=dict(
				n_clusters=optuna.distributions.IntDistribution(2, 30),
			)
		),
	]
)
default_values = [
    dict(
        cohirf_kwargs=[
			dict(
				n_features=0.5,
				repetitions=3,
				base_model_kwargs=dict(
					n_clusters=8,
				)
			),
			dict(
				n_features=0.5,
				repetitions=3,
				base_model_kwargs=dict(
					n_clusters=8,
				)
			),
		]
	),
]
seeds = [i for i in range(5)]
metrics = []
hpo_metric = "adjusted_rand_mean"
for seed in seeds:
    experiment = HPOOpenmlCoClusteringExperiment(
        hpo_seed=seed,
        hpo_metric=hpo_metric,
        direction="maximize",
        search_space=search_space,
        default_values=default_values,
        model=model,
        model_params=model_params,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### experiment parameters
        experiment_name=model.__name__,
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %% [markdown]
# ## 2 stage hpo

# %%
model_stage_1 = BaseCoHiRF
model_params_1 = dict(
    cohirf_kwargs=dict(base_model=KMeans),
)
search_space_stage_1 = dict(
    n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
    repetitions=optuna.distributions.IntDistribution(2, 10),
    base_model_kwargs=dict(
        n_clusters=optuna.distributions.IntDistribution(2, 30),
    ),
)
default_values_stage_1 = [
    dict(
        n_features=0.6,
        repetitions=5,
        base_model_kwargs=dict(
            n_clusters=3,
        ),
    )
]
model_stage_2 = VeCoHiRF
model_params_2 = dict(
    cohirf_model=BaseCoHiRF,
    cohirf_kwargs_shared=dict(),
)
search_space_stage_2 = dict(
    cohirf_kwargs_shared=dict(repetitions=optuna.distributions.IntDistribution(2, 10)),
)
default_values_stage_2 = [
    dict(
        cohirf_kwargs_shared=dict(repetitions=5),
    )
]
hpo_metric_2 = "adjusted_rand_mean"
seeds = [i for i in range(5)]
metrics = []
for seed in seeds:
    experiment = HPOOpenmlVeCoHiRFExperiment(
        hpo_seed=seed,
		hpo_metric_1="adjusted_rand",
		hpo_metric_2=hpo_metric_2,
		direction_1="maximize",
		direction_2="maximize",
		search_space_1=search_space_stage_1,
		search_space_2=search_space_stage_2,
		default_values_1=default_values_stage_1,
		default_values_2=default_values_stage_2,
		model_1=model_stage_1,
		model_2=model_stage_2,
		model_params_1=model_params_1,
		model_params_2=model_params_2,
		### dataset parameters
		dataset_id=61,
		seed_dataset_order=seed,
		standardize=True,
		### coclustering parameters
		n_agents=2,
		p_overlap=0.2,
		max_overlap=0.3,
		### experiment parameters
		experiment_name="test_hpo_twostage",
		log_dir="/home/belucci/code/cocohirf/results/test/logs",
		mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
		raise_on_error=True,
		# calculate_metrics_even_if_too_many_clusters=True,
		check_if_exists=False,
        verbose=1,
	)
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric_2}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric_2}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
model_stage_1 = BaseCoHiRF
model_params_1 = dict(
    cohirf_kwargs=dict(base_model=KMeans),
)
search_space_stage_1 = dict(
    n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
    repetitions=optuna.distributions.IntDistribution(2, 10),
    base_model_kwargs=dict(
        n_clusters=optuna.distributions.IntDistribution(2, 30),
    ),
)
default_values_stage_1 = [
    dict(
        n_features=0.6,
        repetitions=5,
        base_model_kwargs=dict(
            n_clusters=3,
        ),
    )
]
model_stage_2 = VeCoHiRF
model_params_2 = dict(
    cohirf_model=BaseCoHiRF,
    cohirf_kwargs_shared=dict(),
)
search_space_stage_2 = dict(
    cohirf_kwargs_shared=dict(repetitions=optuna.distributions.IntDistribution(2, 10)),
)
default_values_stage_2 = [
    dict(
        cohirf_kwargs_shared=dict(repetitions=5),
    )
]
hpo_metric_2 = "adjusted_rand_mean"
seeds = [i for i in range(5)]
metrics = []
for seed in seeds:
    experiment = HPOOpenmlVeCoHiRFExperiment(
        hpo_seed=seed,
        hpo_metric_1="adjusted_rand",
        hpo_metric_2=hpo_metric_2,
        direction_1="maximize",
        direction_2="maximize",
        search_space_1=search_space_stage_1,
        search_space_2=search_space_stage_2,
        default_values_1=default_values_stage_1,
        default_values_2=default_values_stage_2,
        model_1=model_stage_1,
        model_2=model_stage_2,
        model_params_1=model_params_1,
        model_params_2=model_params_2,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### coclustering parameters
        n_agents=2,
        p_overlap=0.2,
        max_overlap=0.3,
        n_top_trials=1,
        ### experiment parameters
        experiment_name="test_hpo_twostage",
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric_2}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric_2}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
model_stage_1 = BaseCoHiRF
model_params_1 = dict(
    cohirf_kwargs=dict(base_model=KMeans),
)
search_space_stage_1 = dict(
    n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
    repetitions=optuna.distributions.IntDistribution(2, 10),
    base_model_kwargs=dict(
        n_clusters=optuna.distributions.IntDistribution(2, 30),
    ),
)
default_values_stage_1 = [
    dict(
        n_features=0.6,
        repetitions=5,
        base_model_kwargs=dict(
            n_clusters=3,
        ),
    )
]
model_stage_2 = VeCoHiRF
model_params_2 = dict(
    cohirf_model=BaseCoHiRF,
    cohirf_kwargs_shared=dict(),
)
search_space_stage_2 = dict(
    cohirf_kwargs_shared=dict(random_state=optuna.distributions.IntDistribution(0, int(1e6))),
)
default_values_stage_2 = []
hpo_metric_2 = "adjusted_rand_mean"
seeds = [i for i in range(5)]
metrics = []
for seed in seeds:
    experiment = HPOOpenmlVeCoHiRFExperiment(
        hpo_seed=seed,
        hpo_metric_1="adjusted_rand",
        hpo_metric_2=hpo_metric_2,
        direction_1="maximize",
        direction_2="maximize",
        search_space_1=search_space_stage_1,
        search_space_2=search_space_stage_2,
        default_values_1=default_values_stage_1,
        default_values_2=default_values_stage_2,
        model_1=model_stage_1,
        model_2=model_stage_2,
        model_params_1=model_params_1,
        model_params_2=model_params_2,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### coclustering parameters
        n_agents=2,
        p_overlap=0.2,
        max_overlap=0.3,
        n_top_trials=5,
        ### experiment parameters
        experiment_name="test_hpo_twostage",
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric_2}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric_2}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
model_stage_1 = BaseCoHiRF
model_params_1 = dict(
    cohirf_kwargs=dict(base_model=KMeans),
)
search_space_stage_1 = dict(
    n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
    repetitions=optuna.distributions.IntDistribution(2, 10),
    base_model_kwargs=dict(
        n_clusters=optuna.distributions.IntDistribution(2, 30),
    ),
)
default_values_stage_1 = [
    dict(
        n_features=0.6,
        repetitions=5,
        base_model_kwargs=dict(
            n_clusters=3,
        ),
    )
]
model_stage_2 = VeCoHiRF
model_params_2 = dict(
    cohirf_model=BaseCoHiRF,
    cohirf_kwargs_shared=dict(),
)
search_space_stage_2 = dict(
    cohirf_kwargs_shared=dict(
        repetitions=optuna.distributions.IntDistribution(2, 10),
        random_state=optuna.distributions.IntDistribution(0, int(1e6)),
    ),
)
default_values_stage_2 = [
    dict(
        cohirf_kwargs_shared=dict(repetitions=5, random_state=0),
    )
]
hpo_metric_2 = "adjusted_rand_mean"
seeds = [i for i in range(5)]
metrics = []
for seed in seeds:
    experiment = HPOOpenmlVeCoHiRFExperiment(
        hpo_seed=seed,
        hpo_metric_1="adjusted_rand",
        hpo_metric_2=hpo_metric_2,
        direction_1="maximize",
        direction_2="maximize",
        search_space_1=search_space_stage_1,
        search_space_2=search_space_stage_2,
        default_values_1=default_values_stage_1,
        default_values_2=default_values_stage_2,
        model_1=model_stage_1,
        model_2=model_stage_2,
        model_params_1=model_params_1,
        model_params_2=model_params_2,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### coclustering parameters
        n_agents=2,
        p_overlap=0.2,
        max_overlap=0.3,
        n_top_trials=5,
        ### experiment parameters
        experiment_name="test_hpo_twostage",
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric_2}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric_2}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
model_stage_1 = BaseCoHiRF
model_params_1 = dict(
    cohirf_kwargs=dict(base_model=KMeans),
)
search_space_stage_1 = dict(
    n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
    repetitions=optuna.distributions.IntDistribution(2, 10),
    base_model_kwargs=dict(
        n_clusters=optuna.distributions.IntDistribution(2, 30),
    ),
)
default_values_stage_1 = [
    dict(
        n_features=0.6,
        repetitions=5,
        base_model_kwargs=dict(
            n_clusters=3,
        ),
    )
]
model_stage_2 = VeCoHiRF
model_params_2 = dict(
    cohirf_model=BaseCoHiRF,
    cohirf_kwargs_shared=dict(),
)
search_space_stage_2 = dict(
    cohirf_kwargs_shared=dict(random_state=optuna.distributions.IntDistribution(0, int(1e6))),
)
default_values_stage_2 = []
hpo_metric_2 = "adjusted_rand_mean"
seeds = [i for i in range(5)]
metrics = []
for seed in seeds:
    experiment = HPOOpenmlVeCoHiRFExperiment(
        hpo_seed=seed,
        hpo_metric_1="adjusted_rand",
        hpo_metric_2=hpo_metric_2,
        direction_1="maximize",
        direction_2="maximize",
        search_space_1=search_space_stage_1,
        search_space_2=search_space_stage_2,
        default_values_1=default_values_stage_1,
        default_values_2=default_values_stage_2,
        model_1=model_stage_1,
        model_2=model_stage_2,
        model_params_1=model_params_1,
        model_params_2=model_params_2,
        n_trials_1=50,
        n_trials_2=30,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### coclustering parameters
        n_agents=2,
        p_overlap=0.2,
        max_overlap=0.3,
        n_top_trials=5,
        ### experiment parameters
        experiment_name="test_hpo_twostage",
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric_2}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric_2}: {mean_metric:.4f} +/- {std_metric:.4f}")

# %%
model_stage_1 = BaseCoHiRF
model_params_1 = dict(
    cohirf_kwargs=dict(base_model=KMeans),
)
search_space_stage_1 = dict(
    n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
    repetitions=optuna.distributions.IntDistribution(2, 10),
    base_model_kwargs=dict(
        n_clusters=optuna.distributions.IntDistribution(2, 30),
    ),
)
default_values_stage_1 = [
    dict(
        n_features=0.6,
        repetitions=5,
        base_model_kwargs=dict(
            n_clusters=3,
        ),
    )
]
model_stage_2 = VeCoHiRF
model_params_2 = dict(
    cohirf_model=BaseCoHiRF,
    cohirf_kwargs_shared=dict(),
)
search_space_stage_2 = dict(
    # cohirf_kwargs_shared=dict(random_state=optuna.distributions.IntDistribution(0, int(1e6))),
)
default_values_stage_2 = []
hpo_metric_2 = "adjusted_rand_mean"
seeds = [i for i in range(5)]
metrics = []
for seed in seeds:
    experiment = HPOOpenmlVeCoHiRFExperiment(
        hpo_seed=seed,
        hpo_metric_1="adjusted_rand",
        hpo_metric_2=hpo_metric_2,
        direction_1="maximize",
        direction_2="maximize",
        search_space_1=search_space_stage_1,
        search_space_2=search_space_stage_2,
        default_values_1=default_values_stage_1,
        default_values_2=default_values_stage_2,
        model_1=model_stage_1,
        model_2=model_stage_2,
        model_params_1=model_params_1,
        model_params_2=model_params_2,
        n_trials_1=50,
        n_trials_2=30,
        ### dataset parameters
        dataset_id=61,
        seed_dataset_order=seed,
        standardize=True,
        ### coclustering parameters
        n_agents=2,
        p_overlap=0.2,
        max_overlap=0.3,
        n_top_trials=5,
        ### experiment parameters
        experiment_name="test_hpo_twostage",
        log_dir="/home/belucci/code/cocohirf/results/test/logs",
        mlflow_tracking_uri="sqlite:////home/belucci/code/cocohirf/results/test/mlflow.db",
        raise_on_error=True,
        # calculate_metrics_even_if_too_many_clusters=True,
        check_if_exists=False,
        verbose=1,
    )
    results = experiment.run(return_results=True)[0]
    metric = results["evaluate_model_return"][f"best/{hpo_metric_2}"]
    metrics.append(metric)

clear_output(wait=True)
mean_metric = np.mean(metrics)
std_metric = np.std(metrics)
print(f"Mean {hpo_metric_2}: {mean_metric:.4f} +/- {std_metric:.4f}")
