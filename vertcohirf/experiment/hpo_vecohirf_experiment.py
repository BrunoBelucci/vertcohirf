from typing import Optional
from abc import ABC, abstractmethod
from functools import partial
from copy import deepcopy

import numpy as np
import optuna
from optuna import Study, Trial

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from sklearn.base import BaseEstimator

from ml_experiments.base_experiment import BaseExperiment
from ml_experiments.utils import profile_time, profile_memory, flatten_any, update_recursively, unflatten_any
from ml_experiments.tuners import OptunaTuner

from cohirf.experiment.clustering_experiment import ClusteringExperiment
from vertcohirf.experiment.coclustering_experiment import CoClusteringExperiment
from vertcohirf.experiment.tested_models import two_stage_models_dict
import json


def get_trial_fn(study: Study, search_space: dict, random_generator: np.random.Generator, **kwargs):
    seed_model = int(random_generator.integers(0, 2**31 - 1))
    flatten_search_space = flatten_any(search_space)
    trial = study.ask(flatten_search_space)
    trial.set_user_attr("seed_model", seed_model)
    return trial


def training_fn_1(
    trial: Trial,
    features_groups: list,
    agent_i: int,
    stage_1_experiment: type[BaseExperiment],
    model,
    model_params,
    dataset_parameters: dict,
    clustering_parameters: dict,
    experiment_parameters: dict,
    mlflow_run_id: str | None = None,
):
    model_params = model_params.copy()

    # class PartialFeaturesExperiment(stage_1_experiment):
    #     def _load_data(self, *args, **kwargs):
    #         load_data_return = super()._load_data(*args, **kwargs)
    #         X = load_data_return.get("X")
    #         load_data_return["X"] = X.iloc[:, features]
    #         return load_data_return

    trial_number = trial.number
    seed_model = trial.user_attrs["seed_model"]
    trial_model_params = trial.params
    model_params = flatten_any(model_params)
    model_params = update_recursively(model_params, trial_model_params)
    model_params = unflatten_any(model_params)
    # we save the model_params to be used in the second stage
    trial.set_user_attr("model_params", model_params)

    if mlflow_run_id is not None:
        mlflow_run = mlflow.get_run(mlflow_run_id)
        run_first_stage_id = mlflow_run.data.tags[f"child_run_id_first_stage_agent_{agent_i}"]
        run_first_stage = mlflow.get_run(run_first_stage_id)
        run_trial_id = run_first_stage.data.tags[f"child_run_id_{trial_number}"]
        trial.set_user_attr("child_run_id", run_trial_id)
    else:
        run_trial_id = None

    experiment = stage_1_experiment(
        mlflow_run_id=run_trial_id,
        agent_i=agent_i,
        features_groups=features_groups,
        # dataset parameters
        **dataset_parameters,
        # clustering parameters
        model=model,
        model_params=model_params,
        seed_model=seed_model,
        **clustering_parameters,
        # experiment parameters
        **experiment_parameters,
    )
    results: dict = experiment.run(return_results=True)[0]
    if not isinstance(results, dict):
        results = dict()

    if "evaluate_model_return" not in results:  # maybe we already have the run this run and we are getting the stored run result
        keep_results = {metric[len("metrics."):]: value for metric, value in results.items() if metric.startswith("metrics.")}
    else:
        keep_results = results.get("evaluate_model_return", {})
    if "fit_model_return" not in results:
        fit_model_return_elapsed_time = results.get("metrics.fit_model_return_elapsed_time", 0)
    else:
        fit_model_return_elapsed_time = results.get("fit_model_return", {}).get("elapsed_time", 0)
    keep_results["elapsed_time"] = fit_model_return_elapsed_time
    keep_results["seed_model"] = seed_model

    if mlflow_run_id is not None:
        log_metrics = keep_results.copy()
        log_metrics.pop("elapsed_time", None)
        log_metrics.pop("max_memory_used", None)
        mlflow.log_metrics(log_metrics, run_id=run_first_stage_id, step=trial.number)
    return keep_results


def training_fn_2(
    trial: Trial,
    features_groups: list,
    studies: list,
    stage_2_experiment: type[BaseExperiment],
    vecohirf_model,
    vecohirf_params,
    dataset_parameters: dict,
    clustering_parameters: dict,
    experiment_parameters: dict,
    mlflow_run_id: str | None = None,
):
    vecohirf_params = vecohirf_params.copy()
    trial_number = trial.number
    seed_model = trial.user_attrs["seed_model"]
    trial_model_params = trial.params.copy()

    # trial_model_params have the best trials for the independent models and the shared parameters between agents
    best_trials_indexes = {
        key: value for key, value in trial_model_params.items() if key.startswith("best_trial_index_")
    }
    # remove best_trials from trial_model_params
    for key in best_trials_indexes.keys():
        trial_model_params.pop(key)
    # order best trials by key to correspond to features_groups order
    ordered_best_trials_indexes = [best_trials_indexes[key] for key in sorted(best_trials_indexes.keys())]

    # define cohirf_kwargs from the best trials
    cohirf_kwargs = []
    for i, best_trial_index in enumerate(ordered_best_trials_indexes):
        best_trial = studies[i].trials[best_trial_index]
        best_model_params = best_trial.user_attrs["model_params"]
        best_model_params["random_state"] = best_trial.user_attrs["seed_model"]
        cohirf_kwargs.append(best_model_params)

    # set model_params
    vecohirf_params = flatten_any(vecohirf_params)
    vecohirf_params = update_recursively(vecohirf_params, trial_model_params)
    vecohirf_params = unflatten_any(vecohirf_params)
    vecohirf_params["cohirf_kwargs"] = cohirf_kwargs

    if mlflow_run_id is not None:
        mlflow_run = mlflow.get_run(mlflow_run_id)
        run_second_stage_id = mlflow_run.data.tags["child_run_id_second_stage"]
        run_second_stage = mlflow.get_run(run_second_stage_id)
        run_trial_id = run_second_stage.data.tags[f"child_run_id_{trial_number}"]
        trial.set_user_attr("child_run_id", run_trial_id)
    else:
        run_trial_id = None

    # NEED TO CHECK WHAT WE ARE DOING WITH SEED OF INNER MODELS
    experiment = stage_2_experiment(
        mlflow_run_id=run_trial_id,
        # dataset parameters
        **dataset_parameters,
        # clustering parameters
        model=vecohirf_model,
        model_params=vecohirf_params,
        seed_model=seed_model,
        **clustering_parameters,
        # coclustering parameters
        features_groups=features_groups,
        # experiment parameters
        **experiment_parameters,
    )
    results: dict = experiment.run(return_results=True)[0]
    if not isinstance(results, dict):
        results = dict()

    if (
        "evaluate_model_return" not in results
    ):  # maybe we already have the run this run and we are getting the stored run result
        keep_results = {
            metric[len("metrics.") :]: value for metric, value in results.items() if metric.startswith("metrics.")
        }
    else:
        keep_results = results.get("evaluate_model_return", {})
    fit_model_return_elapsed_time = results.get("fit_model_return", {}).get("elapsed_time", None)
    keep_results["elapsed_time"] = fit_model_return_elapsed_time
    keep_results["seed_model"] = seed_model

    if mlflow_run_id is not None:
        log_metrics = keep_results.copy()
        log_metrics.pop("elapsed_time", None)
        log_metrics.pop("max_memory_used", None)
        mlflow.log_metrics(log_metrics, run_id=run_second_stage_id, step=trial.number)
    return keep_results


def run_2_stage_hpo(
    clustering_parameters: dict,
    experiment_parameters: dict,
    dataset_parameters: dict,
    cohirf_model,
    vecohirf_model,
    cohirf_params,
    vecohirf_params,
    tunner_1: OptunaTuner,
    tunner_2: OptunaTuner,
    cohirf_search_space: dict,
    vecohirf_search_space: dict,
    cohirf_default_values: list,
    vecohirf_default_values: list,
    n_top_trials: int,
    metric_1: str,
    metric_2: str,
    direction_1: str,
    direction_2: str,
    stage_1_experiment: type[ClusteringExperiment],
    stage_2_experiment: type[CoClusteringExperiment],
    features_groups: list,
    random_generator: np.random.Generator,
    mlflow_tracking_uri: str | None = None,
    mlflow_run_id: str | None = None,
) -> tuple[list[Study], Study]:

    if mlflow_run_id is not None:
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=mlflow_tracking_uri)
        mlflow_run = mlflow.get_run(mlflow_run_id)
        run_first_stage_ids = [
            mlflow_run.data.tags[f"child_run_id_first_stage_agent_{i_agent}"] for i_agent in range(len(features_groups))
        ]
        run_second_stage_id = mlflow_run.data.tags["child_run_id_second_stage"]
    else:
        mlflow_client = None
        run_first_stage_ids = None
        run_second_stage_id = None

    get_trial_fn_partial = partial(get_trial_fn, random_generator=random_generator)

    studies = []
    for agent_i in range(len(features_groups)):
        if isinstance(cohirf_model, list):
            cohirf_model_agent = cohirf_model[agent_i]
        else:
            cohirf_model_agent = cohirf_model

        if isinstance(cohirf_params, list):
            cohirf_params_agent = cohirf_params[agent_i]
        else:
            cohirf_params_agent = cohirf_params

        if isinstance(cohirf_search_space, list):
            cohirf_search_space_agent = cohirf_search_space[agent_i]
        else:
            cohirf_search_space_agent = cohirf_search_space

        if isinstance(cohirf_default_values, list) and len(cohirf_default_values) > 0 and isinstance(cohirf_default_values[0], list):
            cohirf_default_values_agent = cohirf_default_values[agent_i]
        else:
            cohirf_default_values_agent = cohirf_default_values

        training_fn_1_partial = partial(
            training_fn_1,
            features_groups=features_groups,
            mlflow_run_id=mlflow_run_id,
            agent_i=agent_i,
            stage_1_experiment=stage_1_experiment,
            model=cohirf_model_agent,
            model_params=cohirf_params_agent,
            dataset_parameters=dataset_parameters,
            clustering_parameters=clustering_parameters,
            experiment_parameters=experiment_parameters,
        )
        study_1 = tunner_1.tune(
            training_fn=training_fn_1_partial,
            search_space=cohirf_search_space_agent,
            direction=direction_1,
            metric=metric_1,
            enqueue_configurations=cohirf_default_values_agent,
            get_trial_fn=get_trial_fn_partial,
        )
        best_value = study_1.best_value
        if best_value == np.inf or best_value == -np.inf:
            message = f"No best value found for agent {i}, it may be that all trials failed."
            # set as failed all runs of first stage for this agent and subsequent agents
            if mlflow_run_id is not None:
                for j in range(agent_i, len(features_groups)):
                    run_first_stage_id = run_first_stage_ids[j]
                    mlflow_client.set_terminated(run_first_stage_id, status="FAILED")
                    mlflow_client.set_tag(run_first_stage_id, "raised_exception", "True")
                    mlflow_client.set_tag(run_first_stage_id, "EXCEPTION", message)
            raise ValueError(message)
        else:
            # terminate run
            if mlflow_run_id is not None:
                run_first_stage_id = run_first_stage_ids[agent_i]
                best_trial = study_1.best_trial

                params_to_log = {f'best/{param}': value for param, value in best_trial.params.items()}
                best_child_run_id = best_trial.user_attrs.get('child_run_id', None)
                params_to_log['best/child_run_id'] = best_child_run_id
                mlflow.log_params(params_to_log, run_id=run_first_stage_id)

                best_trial_result = best_trial.user_attrs.get("result", dict())
                best_metric_results = {f"best/{metric}": value for metric, value in best_trial_result.items()}
                best_value = best_trial.value
                best_metric_results["best/value"] = best_value
                mlflow.log_metrics(best_metric_results, run_id=run_first_stage_id)

                mlflow_client.set_terminated(run_first_stage_id, status="FINISHED")
                mlflow_client.set_tag(run_first_stage_id, "raised_exception", "False")
        studies.append(study_1)

    top_n_trials = {}
    for i, study in enumerate(studies):
        trials = study.trials
        trials_indexes = list(range(len(trials)))
        sorted_trials_indexes = sorted(trials_indexes, key=lambda i: trials[i].value)
        if direction_1 == "maximize":
            sorted_trials_indexes = sorted_trials_indexes[::-1]
        top_n_trials[f"best_trial_index_{i}"] = optuna.distributions.CategoricalDistribution(
            sorted_trials_indexes[:n_top_trials]
        )

    training_fn_2_partial = partial(
        training_fn_2,
        features_groups=features_groups,
        studies=studies,
        stage_2_experiment=stage_2_experiment,
        mlflow_run_id=mlflow_run_id,
        vecohirf_model=vecohirf_model,
        vecohirf_params=vecohirf_params,
        dataset_parameters=dataset_parameters,
        clustering_parameters=clustering_parameters,
        experiment_parameters=experiment_parameters,
    )

    vecohirf_search_space.update(top_n_trials)

    study_2 = tunner_2.tune(
        training_fn=training_fn_2_partial,
        search_space=vecohirf_search_space,
        direction=direction_2,
        metric=metric_2,
        enqueue_configurations=vecohirf_default_values,
        get_trial_fn=get_trial_fn_partial,
    )

    best_value = study_2.best_value
    if best_value == np.inf or best_value == -np.inf:
        message = f"No best value found for second stage, it may be that all trials failed."
        if mlflow_run_id is not None:
            mlflow_client.set_terminated(run_second_stage_id, status="FAILED")
            mlflow_client.set_tag(run_second_stage_id, "raised_exception", "True")
            mlflow_client.set_tag(run_second_stage_id, "EXCEPTION", message)
        raise ValueError(message)
    else:
        if mlflow_run_id is not None:
            best_trial = study_2.best_trial

            params_to_log = {f'best/{param}': value for param, value in best_trial.params.items()}
            best_child_run_id = best_trial.user_attrs.get('child_run_id', None)
            params_to_log['best/child_run_id'] = best_child_run_id
            mlflow.log_params(params_to_log, run_id=run_second_stage_id)

            best_trial_result = best_trial.user_attrs.get("result", dict())
            best_metric_results = {f"best/{metric}": value for metric, value in best_trial_result.items()}
            best_value = best_trial.value
            best_metric_results["best/value"] = best_value
            mlflow.log_metrics(best_metric_results, run_id=run_second_stage_id)

            mlflow_client.set_terminated(run_second_stage_id, status="FINISHED")
            mlflow_client.set_tag(run_second_stage_id, "raised_exception", "False")
    return studies, study_2


class HPOVeCoHiRFExperiment(BaseExperiment, ABC):
    """Basically HPOExperiment but it is valid only for VeCoHiRF, where we are performing a 2-stage HPO.
    
    In the first stage we perform HPO for each agent independently with a CoHiRF model, in the second stage we perform 
    HPO for the VeCoHiRF model using the best CoHiRF models found in the first stage as base models and maybe tuning
    shared hyperparameters.
    """

    def __init__(
        self,
        *args,
        hpo_framework: str = "optuna",
        # general
        n_trials_1: int = 50,
        n_trials_2: int = 30,
        timeout_hpo_1: int = 0,
        timeout_hpo_2: int = 0,
        timeout_trial_1: int = 0,
        timeout_trial_2: int = 0,
        max_concurrent_trials_1: int = 1,
        max_concurrent_trials_2: int = 1,
        hpo_seed: int = 0,
        # optuna
        sampler_1: str = "tpe",
        sampler_2: str = "tpe",
        pruner_1: str = "none",
        pruner_2: str = "none",
        hpo_metric_1: str = "adjusted_rand",
        hpo_metric_2: str = "adjusted_rand_mean",
        direction_1: str = "maximize",
        direction_2: str = "maximize",
        model_alias: Optional[str] = None,  # model alias to load from tested_models
        cohirf_model: Optional[BaseEstimator | type[BaseEstimator] | list] = None,
        vecohirf_model: Optional[BaseEstimator | type[BaseEstimator]] = None,
        cohirf_params: Optional[dict | list[dict]] = None,
        vecohirf_params: Optional[dict] = None,
        cohirf_search_space: Optional[dict | list[dict]] = None,
        vecohirf_search_space: Optional[dict] = None,
        cohirf_default_values: Optional[list | list[list]] = None,
        vecohirf_default_values: Optional[list] = None,
        n_top_trials: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hpo_framework = hpo_framework
        # general
        self.n_trials_1 = n_trials_1
        self.n_trials_2 = n_trials_2
        self.timeout_hpo_1 = timeout_hpo_1
        self.timeout_hpo_2 = timeout_hpo_2
        self.timeout_trial_1 = timeout_trial_1
        self.timeout_trial_2 = timeout_trial_2
        self.max_concurrent_trials_1 = max_concurrent_trials_1
        self.max_concurrent_trials_2 = max_concurrent_trials_2
        self.hpo_seed = hpo_seed
        # optuna
        self.sampler_1 = sampler_1
        self.sampler_2 = sampler_2
        self.pruner_1 = pruner_1
        self.pruner_2 = pruner_2
        self.direction_1 = direction_1
        self.direction_2 = direction_2
        self.hpo_metric_1 = hpo_metric_1
        self.hpo_metric_2 = hpo_metric_2
        self.cohirf_search_space = cohirf_search_space
        self.vecohirf_search_space = vecohirf_search_space
        self.cohirf_default_values = cohirf_default_values
        self.vecohirf_default_values = vecohirf_default_values
        self.cohirf_model = cohirf_model
        self.vecohirf_model = vecohirf_model
        self.cohirf_params = cohirf_params if cohirf_params is not None else {}
        self.vecohirf_params = vecohirf_params if vecohirf_params is not None else {}
        self.model_alias = model_alias
        self.n_top_trials = n_top_trials

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not issubclass(cls, ClusteringExperiment):
            raise TypeError(f"{cls.__name__} must inherit from {ClusteringExperiment.__name__}")

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        if self.parser is None:
            raise ValueError('Parser is not initialized, please call the constructor of the class first')
        self.parser.add_argument('--hpo_framework', type=str, default=self.hpo_framework)
        # general
        self.parser.add_argument('--n_trials_1', type=int, default=self.n_trials_1)
        self.parser.add_argument('--n_trials_2', type=int, default=self.n_trials_2)
        self.parser.add_argument('--timeout_hpo_1', type=int, default=self.timeout_hpo_1)
        self.parser.add_argument('--timeout_hpo_2', type=int, default=self.timeout_hpo_2)
        self.parser.add_argument('--timeout_trial_1', type=int, default=self.timeout_trial_1)
        self.parser.add_argument('--timeout_trial_2', type=int, default=self.timeout_trial_2)
        self.parser.add_argument('--max_concurrent_trials_1', type=int, default=self.max_concurrent_trials_1)
        self.parser.add_argument('--max_concurrent_trials_2', type=int, default=self.max_concurrent_trials_2)
        self.parser.add_argument('--hpo_seed', type=int, default=self.hpo_seed)
        # optuna
        self.parser.add_argument('--sampler_1', type=str, default=self.sampler_1)
        self.parser.add_argument('--sampler_2', type=str, default=self.sampler_2)
        self.parser.add_argument('--pruner_1', type=str, default=self.pruner_1)
        self.parser.add_argument('--pruner_2', type=str, default=self.pruner_2)
        self.parser.add_argument('--direction_1', type=str, default=self.direction_1)
        self.parser.add_argument('--direction_2', type=str, default=self.direction_2)
        self.parser.add_argument('--hpo_metric_1', type=str, default=self.hpo_metric_1)
        self.parser.add_argument('--hpo_metric_2', type=str, default=self.hpo_metric_2)
        self.parser.add_argument("--model_alias", type=str, default=self.model_alias)
        self.parser.add_argument('--n_top_trials', type=int, default=self.n_top_trials)
        self.parser.add_argument(
            "--cohirf_params", type=json.loads, default=self.cohirf_params, help="Parameters for the model."
        )
        self.parser.add_argument(
            "--vecohirf_params", type=json.loads, default=self.vecohirf_params, help="Parameters for the model."
        )

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.hpo_framework = args.hpo_framework
        # general
        self.n_trials_1 = args.n_trials_1
        self.n_trials_2 = args.n_trials_2
        self.timeout_hpo_1 = args.timeout_hpo_1
        self.timeout_hpo_2 = args.timeout_hpo_2
        self.timeout_trial_1 = args.timeout_trial_1
        self.timeout_trial_2 = args.timeout_trial_2
        self.max_concurrent_trials_1 = args.max_concurrent_trials_1
        self.max_concurrent_trials_2 = args.max_concurrent_trials_2
        self.hpo_seed = args.hpo_seed
        # optuna
        self.sampler_1 = args.sampler_1
        self.sampler_2 = args.sampler_2
        self.pruner_1 = args.pruner_1
        self.pruner_2 = args.pruner_2
        self.direction_1 = args.direction_1
        self.direction_2 = args.direction_2
        self.hpo_metric_1 = args.hpo_metric_1
        self.hpo_metric_2 = args.hpo_metric_2
        self.model_alias = args.model_alias
        self.n_top_trials = args.n_top_trials
        self.cohirf_params = args.cohirf_params
        self.vecohirf_params = args.vecohirf_params
        return args

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params.update(
            {
                "hpo_framework": self.hpo_framework,
                "n_trials_1": self.n_trials_1,
                "n_trials_2": self.n_trials_2,
                "timeout_hpo_1": self.timeout_hpo_1,
                "timeout_hpo_2": self.timeout_hpo_2,
                "timeout_trial_1": self.timeout_trial_1,
                "timeout_trial_2": self.timeout_trial_2,
                "max_concurrent_trials_1": self.max_concurrent_trials_1,
                "max_concurrent_trials_2": self.max_concurrent_trials_2,
                "hpo_seed": self.hpo_seed,
                "sampler_1": self.sampler_1,
                "sampler_2": self.sampler_2,
                "pruner_1": self.pruner_1,
                "pruner_2": self.pruner_2,
                "direction_1": self.direction_1,
                "direction_2": self.direction_2,
                "hpo_metric_1": self.hpo_metric_1,
                "hpo_metric_2": self.hpo_metric_2,
                "model_alias": self.model_alias,
                "n_top_trials": self.n_top_trials,
                "cohirf_params": self.cohirf_params,
                "vecohirf_params": self.vecohirf_params,
            }
        )
        return unique_params

    @property
    def models_dict(self):
        return two_stage_models_dict.copy()

    def _before_fit_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        hpo_seed = unique_params["hpo_seed"]
        ret = super()._before_fit_model(combination, unique_params, extra_params, mlflow_run_id, **kwargs)
        random_generator = np.random.default_rng(hpo_seed)
        ret["random_generator"] = random_generator
        return ret

    def _load_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        model = unique_params["model_alias"]
        cohirf_params = unique_params["cohirf_params"]
        vecohirf_params = unique_params["vecohirf_params"]
        sampler_1 = unique_params["sampler_1"]
        sampler_2 = unique_params["sampler_2"]
        pruner_1 = unique_params["pruner_1"]
        pruner_2 = unique_params["pruner_2"]
        n_trials_1 = unique_params["n_trials_1"]
        n_trials_2 = unique_params["n_trials_2"]
        timeout_hpo_1 = unique_params["timeout_hpo_1"]
        timeout_hpo_2 = unique_params["timeout_hpo_2"]
        timeout_trial_1 = unique_params["timeout_trial_1"]
        timeout_trial_2 = unique_params["timeout_trial_2"]
        max_concurrent_trials_1 = unique_params["max_concurrent_trials_1"]
        max_concurrent_trials_2 = unique_params["max_concurrent_trials_2"]
        hpo_seed = unique_params["hpo_seed"]

        if isinstance(model, str):
            model_dict = deepcopy(self.models_dict[model])
            cohirf_model = model_dict["cohirf_model"]
            cohirf_params = model_dict["cohirf_params"]
            cohirf_params = update_recursively(cohirf_params, cohirf_params)
            cohirf_search_space = model_dict["cohirf_search_space"]
            cohirf_default_values = model_dict["cohirf_default_values"]
            vecohirf_model = model_dict["vecohirf_model"]
            vecohirf_params = model_dict["vecohirf_params"]
            vecohirf_params = update_recursively(vecohirf_params, vecohirf_params)
            vecohirf_search_space = model_dict["vecohirf_search_space"]
            vecohirf_default_values = model_dict["vecohirf_default_values"]
        else:
            if self.cohirf_model is None or self.vecohirf_model is None or self.cohirf_search_space is None or self.vecohirf_search_space is None:
                raise ValueError("If model is not a string, cohirf_model, vecohirf_model, cohirf_search_space and vecohirf_search_space must be provided")
            cohirf_model = self.cohirf_model
            cohirf_search_space = self.cohirf_search_space
            cohirf_default_values = self.cohirf_default_values if self.cohirf_default_values is not None else []
            vecohirf_model = self.vecohirf_model
            vecohirf_search_space = self.vecohirf_search_space
            vecohirf_default_values = self.vecohirf_default_values if self.vecohirf_default_values is not None else []

        tunner_1 = OptunaTuner(
            sampler=sampler_1,
            pruner=pruner_1,
            n_trials=n_trials_1,
            timeout_total=timeout_hpo_1,
            timeout_trial=timeout_trial_1,
            seed=hpo_seed,
        )

        tunner_2 = OptunaTuner(
            sampler=sampler_2,
            pruner=pruner_2,
            n_trials=n_trials_2,
            timeout_total=timeout_hpo_2,
            timeout_trial=timeout_trial_2,
            seed=hpo_seed,
        )

        return dict(
            cohirf_model=cohirf_model,
            cohirf_params=cohirf_params,
            cohirf_search_space=cohirf_search_space,
            cohirf_default_values=cohirf_default_values,
            vecohirf_model=vecohirf_model,
            vecohirf_params=vecohirf_params,
            vecohirf_search_space=vecohirf_search_space,
            vecohirf_default_values=vecohirf_default_values,
            tunner_1=tunner_1,
            tunner_2=tunner_2,
        )

    @abstractmethod
    def get_dataset_parameters(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        # something like
        # dataset_parameters = dict(
        #     dataset_id=combination.get("dataset_id", None),
        #     seed_dataset_order=combination.get("seed_dataset_order", None),
        #     standardize=self.standardize,
        # )
        # return dataset_parameters
        raise NotImplementedError("This method should be implemented in the subclass")

    @property
    @abstractmethod
    def stage_1_experiment(self):
        # something like
        # return OpenmlClusteringExperiment
        raise NotImplementedError("This property should be implemented in the subclass")

    @property
    @abstractmethod
    def stage_2_experiment(self):
        # something like
        # return OpenmlCoClusteringExperiment
        raise NotImplementedError("This property should be implemented in the subclass")

    @profile_time(enable_based_on_attribute="profile_time")
    @profile_memory(enable_based_on_attribute="profile_memory")
    def _fit_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        cohirf_model = kwargs["load_model_return"]["cohirf_model"]
        cohirf_params = kwargs["load_model_return"]["cohirf_params"]
        cohirf_search_space = kwargs["load_model_return"]["cohirf_search_space"]
        cohirf_default_values = kwargs["load_model_return"]["cohirf_default_values"]
        vecohirf_model = kwargs["load_model_return"]["vecohirf_model"]
        vecohirf_params = kwargs["load_model_return"]["vecohirf_params"]
        vecohirf_search_space = kwargs["load_model_return"]["vecohirf_search_space"]
        vecohirf_default_values = kwargs["load_model_return"]["vecohirf_default_values"]
        tunner_1 = kwargs["load_model_return"]["tunner_1"]
        tunner_2 = kwargs["load_model_return"]["tunner_2"]

        dataset_parameters = self.get_dataset_parameters(combination, unique_params, extra_params, mlflow_run_id, **kwargs)

        # check if stage_1_experiment inherits from ClusteringExperiment and stage_2_experiment from CoClusteringExperiment
        stage_1_experiment = self.stage_1_experiment
        if not issubclass(stage_1_experiment, ClusteringExperiment):
            raise TypeError(f"{stage_1_experiment.__name__} must inherit from {ClusteringExperiment.__name__}")
        stage_2_experiment = self.stage_2_experiment
        if not issubclass(stage_2_experiment, CoClusteringExperiment):
            raise TypeError(f"{stage_2_experiment.__name__} must inherit from {CoClusteringExperiment.__name__}")

        clustering_parameters = dict(
            # those parameters will be set according to stage 1 or 2
            # note that we suppose here that the implemented class is a subclass of ClusteringExperiment
            # model=self.model_1,
            # model_params=self.model_params_1,
            # seed_model=None,
            n_jobs=self.n_jobs,
            clean_data_dir=False,
            calculate_davies_bouldin=self.calculate_davies_bouldin,
            chunk_size_davies_bouldin=self.chunk_size_davies_bouldin,
            calculate_full_silhouette=self.calculate_full_silhouette,
            calculate_metrics_even_if_too_many_clusters=self.calculate_metrics_even_if_too_many_clusters,
            max_threads=self.max_threads,
        )

        experiment_parameters = dict(
            experiment_name=self.experiment_name,
            log_dir=self.log_dir,
            log_file_name=self.log_file_name,
            work_root_dir=self.work_root_dir,
            save_root_dir=self.save_root_dir,
            clean_work_dir=self.clean_work_dir,
            raise_on_error=self.raise_on_error,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            check_if_exists=self.check_if_exists,
            profile_memory=self.profile_memory,
            profile_time=self.profile_time,
            verbose=0,
        )

        studies_1, study_2 = run_2_stage_hpo(
            dataset_parameters=dataset_parameters,
            clustering_parameters=clustering_parameters,
            experiment_parameters=experiment_parameters,
            stage_1_experiment=stage_1_experiment,
            stage_2_experiment=stage_2_experiment,
            tunner_1=tunner_1,
            tunner_2=tunner_2,
            cohirf_model=cohirf_model,
            vecohirf_model=vecohirf_model,
            cohirf_params=cohirf_params,
            vecohirf_params=vecohirf_params,
            cohirf_search_space=cohirf_search_space,
            vecohirf_search_space=vecohirf_search_space,
            cohirf_default_values=cohirf_default_values,
            vecohirf_default_values=vecohirf_default_values,
            n_top_trials=self.n_top_trials,
            metric_1=self.hpo_metric_1,
            metric_2=self.hpo_metric_2,
            direction_1=self.direction_1,
            direction_2=self.direction_2,
            features_groups=kwargs["after_load_data_return"]["features_groups"],
            random_generator=kwargs["before_fit_model_return"]["random_generator"],
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            mlflow_run_id=mlflow_run_id,
        )

        return dict(studies_1=studies_1, study=study_2)

    def _evaluate_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        hpo_metric = unique_params["hpo_metric_2"]
        study = kwargs["fit_model_return"]["study"]

        best_trial = study.best_trial
        best_trial_result = best_trial.user_attrs.get("result", dict())
        best_metric_results = {f"best/{metric}": value for metric, value in best_trial_result.items()}
        best_value = best_trial.value
        best_metric_results["best/value"] = best_value
        # if best_metric_results is empty it means that every trial failed, we will raise an exception
        if f"best/{hpo_metric}" not in best_metric_results:
            raise ValueError(
                f"Best metric {hpo_metric} not found in the best trial results, it may be that every trial failed."
            )

        if mlflow_run_id is not None:
            params_to_log = {f"best/{param}": value for param, value in best_trial.params.items()}
            best_child_run_id = best_trial.user_attrs.get("child_run_id", None)
            params_to_log["best/child_run_id"] = best_child_run_id
            mlflow.log_params(params_to_log, run_id=mlflow_run_id)
        return best_metric_results

    def _create_mlflow_run(self, *combination, combination_names: list[str], unique_params: dict, extra_params: dict):
        parent_run_id = super()._create_mlflow_run(
            *combination, combination_names=combination_names, unique_params=unique_params, extra_params=extra_params
        )
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        experiment = mlflow_client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment {self.experiment_name} not found in mlflow")
        experiment_id = experiment.experiment_id
        for agent in range(unique_params["n_agents"]):
            run_first_stage = mlflow_client.create_run(experiment_id, tags={MLFLOW_PARENT_RUN_ID: parent_run_id})
            mlflow_client.set_tag(parent_run_id, f"child_run_id_first_stage_agent_{agent}", run_first_stage.info.run_id)
            mlflow_client.update_run(run_first_stage.info.run_id, status="SCHEDULED")
            mlflow_client.set_tag(run_first_stage.info.run_id, "hpo_stage", "1")
            mlflow_client.set_tag(run_first_stage.info.run_id, "agent", agent)
            for trial in range(self.n_trials_1):
                run = mlflow_client.create_run(experiment_id, tags={MLFLOW_PARENT_RUN_ID: run_first_stage.info.run_id})
                run_id = run.info.run_id
                mlflow_client.set_tag(run_first_stage.info.run_id, f"child_run_id_{trial}", run_id)
                mlflow_client.set_tag(run_id, "hpo_stage", "1")
                mlflow_client.set_tag(run_id, "trial_number", trial)
                mlflow_client.update_run(run_id, status="SCHEDULED")

        run_second_stage = mlflow_client.create_run(experiment_id, tags={MLFLOW_PARENT_RUN_ID: parent_run_id})
        mlflow_client.set_tag(parent_run_id, f"child_run_id_second_stage", run_second_stage.info.run_id)
        mlflow_client.set_tag(run_second_stage.info.run_id, "hpo_stage", "2")
        mlflow_client.update_run(run_second_stage.info.run_id, status="SCHEDULED")
        for trial in range(self.n_trials_2):
            run = mlflow_client.create_run(experiment_id, tags={MLFLOW_PARENT_RUN_ID: run_second_stage.info.run_id})
            run_id = run.info.run_id
            mlflow_client.set_tag(run_second_stage.info.run_id, f"child_run_id_{trial}", run_id)
            mlflow_client.set_tag(run_id, "hpo_stage", "2")
            mlflow_client.set_tag(run_id, "trial_number", trial)
            mlflow_client.update_run(run_id, status="SCHEDULED")

        return parent_run_id
