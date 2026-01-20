from cohirf.experiment.classification_clustering_experiment import ClassificationClusteringExperiment
from cocohirf.experiment.coclustering_experiment import CoClusteringExperiment
from cocohirf.experiment.tested_models import models_dict
from cohirf.experiment.hpo_clustering_experiment import HPOClusteringExperiment
from cocohirf.experiment.hpo_vecohirf_experiment import HPOVeCoHiRFExperiment


class ClassificationCoClusteringExperiment(ClassificationClusteringExperiment, CoClusteringExperiment):

    @property
    def models_dict(self):
        return models_dict.copy()


class HPOClassificationCoClusteringExperiment(HPOClusteringExperiment, ClassificationCoClusteringExperiment):

    def _load_simple_experiment(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        experiment = ClassificationCoClusteringExperiment(
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


class HPOClassificationVeCoHiRFExperiment(HPOVeCoHiRFExperiment, ClassificationCoClusteringExperiment):

    @property
    def stage_1_experiment(self):
        return ClassificationCoClusteringExperiment

    @property
    def stage_2_experiment(self):
        return ClassificationCoClusteringExperiment

    def get_dataset_parameters(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: str | None = None, **kwargs
    ):
        n_samples = combination["n_samples"]
        n_random = combination["n_random"]
        n_classes = combination["n_classes"]
        n_informative = combination["n_informative"]
        n_redundant = combination["n_redundant"]
        n_repeated = combination["n_repeated"]
        n_clusters_per_class = combination["n_clusters_per_class"]
        weights = combination["weights"]
        flip_y = combination["flip_y"]
        class_sep = combination["class_sep"]
        hypercube = combination["hypercube"]
        shift = combination["shift"]
        scale = combination["scale"]
        shuffle = combination["shuffle"]
        seed_dataset = combination["seed_dataset"]
        n_features_dataset = combination["n_features_dataset"]
        pct_random = combination["pct_random"]
        add_outlier = combination["add_outlier"]
        std_random = combination["std_random"]
        dataset_parameters = dict(
            n_samples=n_samples,
			n_random=n_random,
			n_classes=n_classes,
			n_informative=n_informative,
			n_redundant=n_redundant,
			n_repeated=n_repeated,
			n_clusters_per_class=n_clusters_per_class,
			weights=weights,
			flip_y=flip_y,
			class_sep=class_sep,
			hypercube=hypercube,
			shift=shift,
			scale=scale,
			shuffle=shuffle,
			seed_dataset=seed_dataset,
			n_features_dataset=n_features_dataset,
			pct_random=pct_random,
			add_outlier=add_outlier,
			std_random=std_random,
        )
        return dataset_parameters
