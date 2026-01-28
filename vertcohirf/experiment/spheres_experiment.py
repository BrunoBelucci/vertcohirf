from cohirf.experiment.spherical_clustering_experiment import SphericalClusteringExperiment
from vertcohirf.experiment.coclustering_experiment import CoClusteringExperiment
from vertcohirf.experiment.tested_models import models_dict
from cohirf.experiment.hpo_clustering_experiment import HPOClusteringExperiment
from vertcohirf.experiment.hpo_vecohirf_experiment import HPOVeCoHiRFExperiment


class SphericalnCoClusteringExperiment(SphericalClusteringExperiment, CoClusteringExperiment):

    @property
    def models_dict(self):
        return models_dict.copy()


class HPOSphericalCoClusteringExperiment(HPOClusteringExperiment, SphericalnCoClusteringExperiment):

    def _load_simple_experiment(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        experiment = SphericalnCoClusteringExperiment(
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


class HPOSphericalVeCoHiRFExperiment(HPOVeCoHiRFExperiment, SphericalnCoClusteringExperiment):

    @property
    def stage_1_experiment(self):
        return SphericalnCoClusteringExperiment

    @property
    def stage_2_experiment(self):
        return SphericalnCoClusteringExperiment

    def get_dataset_parameters(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        n_samples = combination["n_samples"]
        seed_dataset = combination["seed_dataset"]
        n_spheres = combination["n_spheres"]
        radius_separation = combination["radius_separation"]
        radius_std = combination["radius_std"]
        add_radius_as_feature = unique_params["add_radius_as_feature"]
        dataset_parameters = dict(
            n_samples=n_samples,
            seed_dataset=seed_dataset,
            n_spheres=n_spheres,
            radius_separation=radius_separation,
            radius_std=radius_std,
            add_radius_as_feature=add_radius_as_feature,
        )
        return dataset_parameters
