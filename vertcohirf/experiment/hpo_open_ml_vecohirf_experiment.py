from vertcohirf.experiment.hpo_vecohirf_experiment import HPOVeCoHiRFExperiment
from vertcohirf.experiment.open_ml_coclustering_experiment import OpenmlCoClusteringExperiment


class HPOOpenmlVeCoHiRFExperiment(HPOVeCoHiRFExperiment, OpenmlCoClusteringExperiment):

    @property
    def stage_1_experiment(self):
        return OpenmlCoClusteringExperiment

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


if __name__ == "__main__":
    experiment = HPOOpenmlVeCoHiRFExperiment()
    experiment.run_from_cli()
