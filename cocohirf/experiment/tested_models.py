from cohirf.models.vecohirf import VeCoHiRF
import optuna
from cocohirf.models.coreset_kmeans import CoresetKMeans
from cocohirf.models.distributed_kmeans import DistributedKMeans
from cocohirf.models.dpvfl import DPVFL


models_dict = {
    VeCoHiRF.__name__: (
        VeCoHiRF,
        dict(),
        dict(
            cohirf_kwargs=dict(
                n_features=optuna.distributions.FloatDistribution(0.1, 0.6),
                repetitions=optuna.distributions.IntDistribution(2, 10),
                kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
            )
        ),
        [
            dict(
                cohirf_kwargs=dict(
                    n_features=0.3,
                    repetitions=5,
                    kmeans_n_clusters=3,
                )
            )
        ],
    ),
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
        dict(mode='v2way'),
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
        dict(mode='vpc'),
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
