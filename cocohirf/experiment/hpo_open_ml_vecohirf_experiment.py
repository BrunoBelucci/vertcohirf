from cocohirf.experiment.hpo_vecohirf_experiment import HPOVeCoHiRFExperiment
from cocohirf.experiment.open_ml_coclustering_experiment import OpenmlCoClusteringExperiment
from cohirf.experiment.open_ml_clustering_experiment import OpenmlClusteringExperiment


class HPOOpenmlVeCoHiRFExperiment(HPOVeCoHiRFExperiment, OpenmlCoClusteringExperiment):

    @property
    def stage_1_experiment(self):
        return OpenmlClusteringExperiment

    @property
    def stage_2_experiment(self):
        return OpenmlCoClusteringExperiment

    def get_dataset_parameters(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        dataset_id = combination["dataset_id"]
        seed_dataset_order = combination["seed_dataset_order"]
        standardize = unique_params["standardize"]
        dataset_parameters = dict(
            dataset_id=dataset_id,
            seed_dataset_order=seed_dataset_order,
            standardize=standardize,
        )
        return dataset_parameters
