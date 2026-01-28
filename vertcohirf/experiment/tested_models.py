from cohirf.experiment.open_ml_clustering_experiment import models_dict as cluster_models_dict
from cohirf.models.vecohirf import VeCoHiRF
from cohirf.models.cohirf import CoHiRF
import optuna
from cocohirf.models.coreset_kmeans import CoresetKMeans
from cocohirf.models.distributed_kmeans import DistributedKMeans
from cocohirf.models.dpvfl import DPVFL
from ml_experiments.utils import update_recursively
from copy import deepcopy


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
two_stage_models_dict = dict()

# create VeCoHiRF versions of CoHiRF models
cohirf_models = [model_name for model_name in models_dict.keys() if model_name.startswith("CoHiRF")]
for model_name in cohirf_models:
    model, model_params, search_space, default_values = models_dict[model_name]
    model_1 = deepcopy(model)
    model_params_1 = deepcopy(model_params)
    search_space_1 = deepcopy(search_space)
    default_values_1 = deepcopy(default_values)
    if model_name == CoHiRF.__name__:
        vecohirf_name = VeCoHiRF.__name__
    else:
        model_name_without_cohirf = model_name[len(CoHiRF.__name__) + 1 :]
        vecohirf_name = VeCoHiRF.__name__ + "-" + model_name_without_cohirf
    model_2 = VeCoHiRF
    model_params_2 = dict(cohirf_model=model, cohirf_kwargs_shared=dict())
    search_space_2 = dict(cohirf_kwargs_shared=dict(random_state=optuna.distributions.IntDistribution(0, int(1e6))))
    default_values_2 = []
    two_stage_models_dict[vecohirf_name] = dict(
        cohirf_model=model_1,
        cohirf_params=model_params_1,
        cohirf_search_space=search_space_1,
        cohirf_default_values=default_values_1,
        vecohirf_model=model_2,
        vecohirf_params=model_params_2,
        vecohirf_search_space=search_space_2,
        vecohirf_default_values=default_values_2,
    )
    # single stage VeCoHiRF
    model_params_vecohirf = dict(cohirf_model=deepcopy(model), cohirf_kwargs=deepcopy(model_params))
    search_space_vecohirf = dict(cohirf_kwargs=deepcopy(search_space))
    default_values_vecohirf = [dict(cohirf_kwargs=deepcopy(dv)) for dv in deepcopy(default_values)]
    models_dict[vecohirf_name] = (VeCoHiRF, model_params_vecohirf, search_space_vecohirf, default_values_vecohirf) 

# create 1-iteration VeCoHiRF versions of VeCoHiRF models
vecohirf_models = [model_name for model_name in two_stage_models_dict.keys() if model_name.startswith(VeCoHiRF.__name__)]
for model_name in vecohirf_models:
    model_1 = deepcopy(two_stage_models_dict[model_name]["cohirf_model"])
    model_params_1 = deepcopy(two_stage_models_dict[model_name]["cohirf_params"])
    search_space_1 = deepcopy(two_stage_models_dict[model_name]["cohirf_search_space"])
    default_values_1 = deepcopy(two_stage_models_dict[model_name]["cohirf_default_values"])
    model_2 = deepcopy(two_stage_models_dict[model_name]["vecohirf_model"])
    model_params_2 = deepcopy(two_stage_models_dict[model_name]["vecohirf_params"])
    search_space_2 = deepcopy(two_stage_models_dict[model_name]["vecohirf_search_space"])
    default_values_2 = deepcopy(two_stage_models_dict[model_name]["vecohirf_default_values"])
    model_params_1 = update_recursively(model_params_1, dict(max_iter=1))
    model_params_2 = update_recursively(model_params_2, dict(cohirf_kwargs_shared=dict(max_iter=1)))
    two_stage_models_dict[model_name + "-1iter"] = dict(
        cohirf_model=model_1,
        cohirf_params=model_params_1,
        cohirf_search_space=search_space_1,
        cohirf_default_values=default_values_1,
        vecohirf_model=model_2,
        vecohirf_params=model_params_2,
        vecohirf_search_space=search_space_2,
        vecohirf_default_values=default_values_2,
    )
    # single stage VeCoHiRF
    model, model_params, search_space, default_values = models_dict[model_name]
    model_params = deepcopy(model_params)
    model_params = update_recursively(model_params, dict(cohirf_kwargs=dict(max_iter=1)))
    search_space = deepcopy(search_space)
    default_values = deepcopy(default_values)
    models_dict[model_name + "-1iter"] = (VeCoHiRF, model_params, search_space, default_values)
