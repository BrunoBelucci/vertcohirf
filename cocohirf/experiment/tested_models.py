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
            coreset_size_div=optuna.distributions.IntDistribution(5, 20),
            alpha=optuna.distributions.FloatDistribution(1.0, 5.0),
            kmeans_n_clusters=optuna.distributions.IntDistribution(2, 30),
        ),
        [
            dict(
                coreset_size_div=10,
                alpha=2.0,
                kmeans_n_clusters=8,
            )
        ],
    ),
	"V2way": (
        DPVFL,
        dict(mode='v2way'),
        dict(
            n_clusters=optuna.distributions.IntDistribution(2, 30),
            m_div=optuna.distributions.IntDistribution(5, 20),
            eps=optuna.distributions.FloatDistribution(0.1, 5.0),
        ),
        [
            dict(
                n_clusters=8,
                m_div=10,
                eps=0.5,
            )
        ],
	),
	"VPC": (
        DPVFL,
        dict(mode='vpc'),
        dict(
            n_clusters=optuna.distributions.IntDistribution(2, 30),
            m_div=optuna.distributions.IntDistribution(5, 20),
            eps=optuna.distributions.FloatDistribution(0.1, 5.0),
        ),
        [
            dict(
                n_clusters=8,
                m_div=10,
                eps=0.5,
            )
        ],
    ),
}
