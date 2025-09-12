from cohirf.experiment.open_ml_clustering_experiment import OpenmlClusteringExperiment
from cocohirf.experiment.coclustering_experiment import CoClusteringExperiment
from cocohirf.experiment.tested_models import models_dict


class OpenmlCoClusteringExperiment(OpenmlClusteringExperiment, CoClusteringExperiment):

    @property
    def models_dict(self):
        return models_dict.copy()


if __name__ == "__main__":
    experiment = OpenmlCoClusteringExperiment()
    experiment.run_from_cli()
