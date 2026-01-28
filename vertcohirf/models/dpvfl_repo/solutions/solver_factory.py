from .BasicKmeans import BasicKmeans
from .PrivLSH import PrivLSH

solver_mapping = {
    'basic': BasicKmeans,
    'lsh_clustering': PrivLSH,
}
