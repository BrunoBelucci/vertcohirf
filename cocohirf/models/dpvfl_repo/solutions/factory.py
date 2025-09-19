from .BasicKmeans import BasicKmeans
from .VPrivClustering import VPrivClustering
from .V2way import V2way
from .PrivLSH import PrivLSH

solutions = {
    "kmeans": BasicKmeans,
    "VPC": VPrivClustering,
    "V2way": V2way,
    "lsh_clustering":  PrivLSH
}
