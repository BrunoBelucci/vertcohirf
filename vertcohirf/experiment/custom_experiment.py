from cohirf.experiment.custom_clustering_experiment import CustomClusteringExperiment
from vertcohirf.experiment.coclustering_experiment import CoClusteringExperiment
from vertcohirf.experiment.tested_models import models_dict
from cohirf.experiment.hpo_clustering_experiment import HPOClusteringExperiment
from vertcohirf.experiment.hpo_vecohirf_experiment import HPOVeCoHiRFExperiment
from typing import Optional
import mlflow
from cohirf.experiment.open_ml_clustering_experiment import preprocess


class CustomCoClusteringExperiment(CustomClusteringExperiment, CoClusteringExperiment):
    
    def _load_data(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        dataset_name = unique_params["dataset_name"]
        standardize = unique_params["standardize"]
        seed_dataset_order = combination["seed_dataset_order"]
        X = self.X
        y = self.y
        # try to infer categorical features from data
        cat_features_names = X.select_dtypes(include=['object', 'category']).columns.tolist()
        cont_features_names = X.select_dtypes(include=['number', 'bool']).columns.tolist()
        agent_i = unique_params.get('agent_i', None)
        
        # if multiple y, we suppose y[0] is the global label and y[1:], if any, are local labels
        if isinstance(y, (tuple, list)):
            if agent_i is None:
                y = y[0]
            else:
                y = y[agent_i + 1]
        n_classes = len(y.unique())
        
        # we will preprocess the data always in the same way
        X, y = preprocess(X, y, cat_features_names, cont_features_names, standardize, seed_dataset_order)
        # log to mlflow to facilitate analysis
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        if mlflow_run_id is not None:
            mlflow.log_params({
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': n_classes,
            }, run_id=mlflow_run_id)
        return {
            'X': X,
            'y': y,
            'cat_features_names': cat_features_names,
            'n_classes': n_classes,
            'dataset_name': dataset_name
        }

    @property
    def models_dict(self):
        return models_dict.copy()


class HPOCustomCoClusteringExperiment(HPOClusteringExperiment, CustomCoClusteringExperiment):

    def _load_simple_experiment(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        experiment = CustomCoClusteringExperiment(
            X=extra_params["X"],
            y=extra_params["y"],
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


class HPOCustomVeCoHiRFExperiment(HPOVeCoHiRFExperiment, CustomCoClusteringExperiment):

    @property
    def stage_1_experiment(self):
        return CustomCoClusteringExperiment
    @property
    def stage_2_experiment(self):
        return CustomCoClusteringExperiment

    def get_dataset_parameters(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        dataset_name = unique_params["dataset_name"]
        standardize = unique_params["standardize"]
        seed_dataset_order = combination["seed_dataset_order"]
        X = extra_params["X"]
        y = extra_params["y"]
        dataset_parameters = dict(
            dataset_name=dataset_name,
            standardize=standardize,
            seed_dataset_order=seed_dataset_order,
            X=X,
            y=y,
        )
        return dataset_parameters
