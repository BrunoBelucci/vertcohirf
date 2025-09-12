from cohirf.models.vecohirf import VeCoHiRF
from cohirf.models.clique import Clique
from cohirf.models.irfllrr import IRFLLRR
from cohirf.models.kmeansproj import KMeansProj
from cohirf.models.proclus import Proclus
from cohirf.models.scsrgf import SpectralSubspaceRandomization
from cohirf.models.sklearn import (KMeans, OPTICS, DBSCAN, AgglomerativeClustering, SpectralClustering,
                                                 MeanShift, AffinityPropagation, HDBSCAN)
from cohirf.models.batch_cohirf import BatchCoHiRF
from cohirf.models.pseudo_kernel import PseudoKernelClustering
from cohirf.models.WBMS import WBMS
from sklearn.kernel_approximation import Nystroem, RBFSampler
import optuna


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
}
