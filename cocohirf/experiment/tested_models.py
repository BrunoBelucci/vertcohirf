from cohirf.experiment.open_ml_clustering_experiment import models_dict as cluster_models_dict
from cohirf.models.vecohirf import VeCoHiRF
from cohirf.models.cohirf import BaseCoHiRF
from cohirf.models.scsrgf import SpectralSubspaceRandomization
from sklearn.cluster import KMeans, DBSCAN
from sklearn.kernel_approximation import RBFSampler
import optuna
from cocohirf.models.coreset_kmeans import CoresetKMeans
from cocohirf.models.distributed_kmeans import DistributedKMeans
from cocohirf.models.dpvfl import DPVFL


models_dict = {
    DistributedKMeans.__name__: (
        DistributedKMeans,
        dict(),
        dict(
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                kmeans_n_clusters=8,
            )
        ],
    ),
    DistributedKMeans.__name__ + "Local": (
        DistributedKMeans,
        dict(use_server_labels=False),
        dict(
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                kmeans_n_clusters=8,
            )
        ],
    ),
    CoresetKMeans.__name__: (
        CoresetKMeans,
        dict(),
        dict(
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                kmeans_n_clusters=8,
            )
        ],
    ),
    "V2way": (
        DPVFL,
        dict(mode="v2way"),
        dict(
            n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                n_clusters=8,
            )
        ],
    ),
    "VPC": (
        DPVFL,
        dict(mode="vpc"),
        dict(
            n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                n_clusters=8,
            )
        ],
    ),
}

# Add clustering models from cohirf
models_dict.update(cluster_models_dict)

two_stage_models_dict = {
    VeCoHiRF.__name__: dict(
        model_1=BaseCoHiRF,
        model_params_1=dict(
            base_model=KMeans,
        ),
        search_space_1=dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 1.0),
            repetitions=optuna.distributions.IntDistribution(2, 10),
            base_model_kwargs=dict(
                n_clusters=optuna.distributions.IntDistribution(2, 30),
            ),
        ),
        default_values_1=[
            dict(
                n_features=0.6,
                repetitions=5,
                base_model_kwargs=dict(
                    n_clusters=3,
                ),
            )
        ],
        model_2=VeCoHiRF,
        model_params_2=dict(
            cohirf_model=BaseCoHiRF,
            cohirf_kwargs_shared=dict(),
        ),
        search_space_2=dict(
            cohirf_kwargs_shared=dict(random_state=optuna.distributions.IntDistribution(0, int(1e6))),
        ),
        default_values_2=[],
    ),
    VeCoHiRF.__name__
    + "-DBSCAN": dict(
        model_1=BaseCoHiRF,
        model_params_1=dict(base_model=DBSCAN),
        search_space_1=dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 1),
            repetitions=optuna.distributions.IntDistribution(1, 10),
            base_model_kwargs=dict(
                eps=optuna.distributions.FloatDistribution(1e-1, 10),
                min_samples=optuna.distributions.IntDistribution(2, 50),
            ),
        ),
        default_values_1=[
            dict(
                n_features=0.3,
                repetitions=5,
                base_model_kwargs=dict(
                    eps=0.5,
                    min_samples=5,
                ),
            ),
        ],
        model_2=VeCoHiRF,
        model_params_2=dict(),
        search_space_2=dict(
            cohirf_kwargs_shared=dict(random_state=optuna.distributions.IntDistribution(0, int(1e6))),
        ),
        default_values_2=[],
    ),
    VeCoHiRF.__name__
    + "-KernelRBF": dict(
        model_1=BaseCoHiRF,
        model_params_1=dict(
            base_model=KMeans,
            transform_method=RBFSampler,
            transform_kwargs=dict(n_components=500),
            representative_method="rbf",
        ),
        search_space_1=dict(
            n_features=optuna.distributions.FloatDistribution(0.1, 1),
            repetitions=optuna.distributions.IntDistribution(2, 10),
            base_model_kwargs=dict(
                n_clusters=optuna.distributions.IntDistribution(2, 5),
            ),
            transform_kwargs=dict(
                gamma=optuna.distributions.FloatDistribution(0.1, 30),
            ),
        ),
        default_values_1=[
            dict(
                n_features=0.3,
                repetitions=5,
                base_model_kwargs=dict(
                    n_clusters=3,
                ),
                transform_kwargs=dict(
                    gamma=1.0,
                ),
            )
        ],
        model_2=VeCoHiRF,
        model_params_2=dict(),
        search_space_2=dict(
            cohirf_kwargs_shared=dict(random_state=optuna.distributions.IntDistribution(0, int(1e6))),
        ),
        default_values_2=[],
    ),
    VeCoHiRF.__name__
    + "-SC-SRGF": dict(
        model_1=SpectralSubspaceRandomization,
        model_params_1=dict(base_model=SpectralSubspaceRandomization, n_features=1.0),
        search_space_1=dict(
            repetitions=optuna.distributions.IntDistribution(2, 10),
            base_model_kwargs=dict(
                n_similarities=optuna.distributions.IntDistribution(10, 30),
                sampling_ratio=optuna.distributions.FloatDistribution(0.2, 0.8),
                sc_n_clusters=optuna.distributions.IntDistribution(2, 5),
            ),
        ),
        default_values_1=[
            dict(
                repetitions=5,
                base_model_kwargs=dict(
                    n_similarities=20,
                    sampling_ratio=0.5,
                    sc_n_clusters=3,
                ),
            )
        ],
        model_2=VeCoHiRF,
        model_params_2=dict(),
        search_space_2=dict(
            cohirf_kwargs_shared=dict(random_state=optuna.distributions.IntDistribution(0, int(1e6))),
        ),
        default_values_2=[],
    ),
}
