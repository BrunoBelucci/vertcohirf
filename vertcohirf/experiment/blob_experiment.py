from cohirf.experiment.blob_clustering_experiment import BlobClusteringExperiment
from vertcohirf.experiment.coclustering_experiment import CoClusteringExperiment
from vertcohirf.experiment.tested_models import models_dict
from cohirf.experiment.hpo_clustering_experiment import HPOClusteringExperiment
from vertcohirf.experiment.hpo_vecohirf_experiment import HPOVeCoHiRFExperiment


class BlobCoClusteringExperiment(BlobClusteringExperiment, CoClusteringExperiment):

    @property
    def models_dict(self):
        return models_dict.copy()


class HPOBlobCoClusteringExperiment(HPOClusteringExperiment, BlobCoClusteringExperiment):

    def _load_simple_experiment(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        experiment = BlobCoClusteringExperiment(
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


class HPOBlobVeCoHiRFExperiment(HPOVeCoHiRFExperiment, BlobCoClusteringExperiment):

    @property
    def stage_1_experiment(self):
        return BlobCoClusteringExperiment

    @property
    def stage_2_experiment(self):
        return BlobCoClusteringExperiment

    def get_dataset_parameters(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        n_samples = combination["n_samples"]
        n_features = combination["n_features_dataset"]
        centers = combination["centers"]
        cluster_std = combination["cluster_std"]
        center_box = combination["center_box"]
        shuffle = combination["shuffle"]
        seed_dataset = combination["seed_dataset"]
        dataset_parameters = dict(
            n_samples=n_samples,
            n_features_dataset=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            shuffle=shuffle,
            seed_dataset=seed_dataset,
        )
        return dataset_parameters
