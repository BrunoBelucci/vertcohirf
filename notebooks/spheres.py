# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: cocohirf
#     language: python
#     name: python3
# ---

# %%
from cocohirf.experiment.spheres_experiment import SphericalnCoClusteringExperiment, HPOSphericalCoClusteringExperiment, HPOSphericalVeCoHiRFExperiment
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ml_experiments.utils import unflatten_any, flatten_any, update_recursively
from cohirf.models.vecohirf import VeCoHiRF
from cocohirf.experiment.tested_models import two_stage_models_dict
import numpy as np
import matplotlib as mpl
from pathlib import Path
from copy import deepcopy

# %%
results_dir = Path("/home/belucci/code/cocohirf/results/hypercube")
results_dir.mkdir(parents=True, exist_ok=True)

# %%
n_samples = 2000
seed_dataset = 0
n_spheres = 2
radius_separation = 0.5
radius_std = 0.01
add_radius_as_feature = False
n_agents = 3
p_overlap = 0.0
sequential_split = True
n_trials = 50
n_trials_1 = n_trials
n_trials_2 = 30
hpo_metric = "adjusted_rand_mean"
direction = "maximize"
features_groups = None

# %%
model = "VeCoHiRF-DBSCAN-1iter"
hpo_experiment = HPOSphericalVeCoHiRFExperiment(
    # dataset
    n_samples=n_samples,
    n_spheres=n_spheres,
    radius_separation=radius_separation,
    radius_std=radius_std,
    add_radius_as_feature=False,
    seed_dataset=seed_dataset,
    n_agents=3,
    p_overlap=1,
    sequential_split=sequential_split,
    features_groups=features_groups,
    # model
    model_alias=model,
    n_trials_1=n_trials_1,
    n_trials_2=n_trials_2,
    # hpo
    hpo_seed=seed_dataset,
    hpo_metric_1=hpo_metric,
    hpo_metric_2=hpo_metric,
    direction_1=direction,
    direction_2=direction,
    calculate_metrics_even_if_too_many_clusters=True,
    # experiment
    raise_on_error=True,
    verbose=1,
)
hpo_result = hpo_experiment.run(return_results=True)[0]
ari = hpo_result["evaluate_model_return"]["best/adjusted_rand_mean"]
print(f"Best Adjusted Rand Index (ARI) from HPO: {ari}")

# %%
model = "VeCoHiRF-DBSCAN-1iter"
hpo_experiment = HPOSphericalCoClusteringExperiment(
    # dataset
    n_samples=n_samples,
    n_spheres=n_spheres,
    radius_separation=radius_separation,
    radius_std=radius_std,
    add_radius_as_feature=False,
    seed_dataset=seed_dataset,
    n_agents=3,
    p_overlap=1,
    sequential_split=sequential_split,
    features_groups=features_groups,
    # model
    model=model,
    n_trials=n_trials,
    # hpo
    hpo_seed=seed_dataset,
    hpo_metric=hpo_metric,
    direction=direction,
    # experiment
    raise_on_error=True,
    verbose=1,
)
hpo_result = hpo_experiment.run(return_results=True)[0]
ari = hpo_result["evaluate_model_return"]["best/adjusted_rand_mean"]
print(f"Best Adjusted Rand Index (ARI) from HPO: {ari}")

# %%
model = "VeCoHiRF-1iter"
hpo_experiment = HPOSphericalVeCoHiRFExperiment(
    # dataset
    n_samples=n_samples,
    n_spheres=n_spheres,
    radius_separation=radius_separation,
    radius_std=radius_std,
    add_radius_as_feature=True,
    seed_dataset=seed_dataset,
    n_agents=3,
    p_overlap=p_overlap,
    sequential_split=sequential_split,
    features_groups=features_groups,
    # model
    model_alias=model,
    n_trials_1=n_trials_1,
    n_trials_2=n_trials_2,
    # hpo
    hpo_seed=seed_dataset,
    hpo_metric_1=hpo_metric,
    hpo_metric_2=hpo_metric,
    direction_1=direction,
    direction_2=direction,
    calculate_metrics_even_if_too_many_clusters=True,
    # experiment
    raise_on_error=True,
    verbose=1,
)
hpo_result = hpo_experiment.run(return_results=True)[0]
ari = hpo_result["evaluate_model_return"]["best/adjusted_rand_mean"]
print(f"Best Adjusted Rand Index (ARI) from HPO: {ari}")

# %%
model = "VeCoHiRF-1iter"
hpo_experiment = HPOSphericalCoClusteringExperiment(
    # dataset
    n_samples=n_samples,
    n_spheres=n_spheres,
    radius_separation=radius_separation,
    radius_std=radius_std,
    add_radius_as_feature=True,
    seed_dataset=seed_dataset,
    n_agents=3,
    p_overlap=p_overlap,
    sequential_split=sequential_split,
    features_groups=features_groups,
    # model
    model=model,
    n_trials=n_trials,
    # hpo
    hpo_seed=seed_dataset,
    hpo_metric=hpo_metric,
    direction=direction,
    # experiment
    raise_on_error=True,
    verbose=1,
)
hpo_result = hpo_experiment.run(return_results=True)[0]
ari = hpo_result["evaluate_model_return"]["best/adjusted_rand_mean"]
print(f"Best Adjusted Rand Index (ARI) from HPO: {ari}")

# %%
vecohirf_model = deepcopy(two_stage_models_dict[model]["model_2"])
model_params = deepcopy(two_stage_models_dict[model]["model_params_2"])
best_params = deepcopy(hpo_result["fit_model_return"]["study"].best_params)
cohirf_kwargs = []
for best_trial_index, index in best_params.items():
    if best_trial_index.startswith("best_trial_index_"):
        agent_idx = int(best_trial_index.split("_")[-1])
        best_trial = deepcopy(hpo_result["fit_model_return"]["studies_1"][agent_idx].trials[index])
        best_model_params = deepcopy(best_trial.user_attrs["model_params"])
        best_model_params["random_state"] = best_trial.user_attrs["seed_model"]
        cohirf_kwargs.append(best_model_params)

# pop best_trial_index_* from best_params
best_params = {k: v for k, v in best_params.items() if not k.startswith("best_trial_index_")}
model_params = flatten_any(model_params)
model_params = update_recursively(model_params, best_params)
model_params = unflatten_any(model_params)
model_params["cohirf_kwargs"] = cohirf_kwargs

# %%
experiment = ClassificationCoClusteringExperiment(
    # dataset
    n_samples=n_samples,
    n_informative=n_informative,
    n_random=n_random,
    n_redundant=n_redundant,
    n_repeated=n_repeated,
    n_classes=n_classes,
    class_sep=class_sep,
    seed_dataset=seed_dataset,
    shift=shift,
    scale=scale,
    n_agents=n_agents,
    p_overlap=p_overlap,
    sequential_split=sequential_split,
    features_groups=features_groups,
	# model
	model=VeCoHiRF,
    model_params=model_params,
    # experiment
    raise_on_error=True,
    verbose=1,
)
result = experiment.run(return_results=True)[0]

# %%
features = result["load_data_return"]["X"]
y_true = result["load_data_return"]["y"]
labels = result["fit_model_return"]["y_pred"]
unique_labels = np.unique(labels)
tsne = TSNE(n_components=2, random_state=seed_dataset)
X_embedded = tsne.fit_transform(features)

# %%
with mpl.rc_context(
    rc={
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times", "serif"],  # Reliable serif fonts
        "mathtext.fontset": "dejavuserif",  # Use DejaVu for math text to avoid missing glyphs
        "axes.unicode_minus": False,  # Use ASCII minus instead of Unicode minus
        "font.size": 12,
        "axes.linewidth": 1.2,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.shadow": False,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "grid.alpha": 0.5,
        "axes.grid": True,
        "grid.linewidth": 0.5,
    }
):
    cm = 1 / 2.54  # centimeters to inches
    fig_scale = 3.0
    fig, ax = plt.subplots(figsize=(12 * fig_scale * cm, 7 * fig_scale * cm))
    # Create discrete colormap with only the colors you need
    n_clusters = len(np.unique(labels))
    colors = plt.colormaps["tab10"].colors
    markers = ["o", "s", "^", "v", "D", "P", "*", "X", "p", "h"]  # circle, square, triangle, etc.
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=[colors[i % len(colors)]],
            marker=markers[i % len(markers)],
            s=90,
            label=f"Cluster {label}",
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_ylabel("t-SNE feature 2")
    ax.set_xlabel("t-SNE feature 1")
    ax.set_title(f"VertCoHiRF: ARI={ari:.3f}")
    fig.savefig(
        results_dir
        / f"hypercube_{model}_n-samples-{n_samples}_n-informative-{n_informative}_n-random-{n_random}_n-classes-{n_classes}_class-sep-{class_sep}_shift-{shift}_scale-{scale}_n-agents-{n_agents}_p-overlap-{p_overlap}_sequential-split-{sequential_split}.pdf",
        dpi=600,
        bbox_inches="tight",
    )

# %%
model = "CoHiRF"
agent_i = [i for i in range(n_agents)]
for ai in agent_i:
    hpo_experiment = HPOSphericalCoClusteringExperiment(
        agent_i=ai,
        # dataset
        n_samples=n_samples,
        n_spheres=n_spheres,
        radius_separation=radius_separation,
        radius_std=radius_std,
        add_radius_as_feature=True,
        seed_dataset=seed_dataset,
        n_agents=3,
        p_overlap=p_overlap,
        sequential_split=sequential_split,
        features_groups=features_groups,
        # model
        model=model,
        n_trials=n_trials,
        # seed_model=seed_dataset,
        # hpo
        hpo_seed=seed_dataset,
        hpo_metric=hpo_metric,
        direction=direction,
        # experiment
        raise_on_error=True,
        verbose=1,
    )
    hpo_result = hpo_experiment.run(return_results=True)[0]
    ari = hpo_result["evaluate_model_return"]["best/adjusted_rand_mean"]
    print(f"Agent {ai} - Best Adjusted Rand Index (ARI) from HPO: {ari}")
    


# %%

    best_params = hpo_result["fit_model_return"]["study"].best_params
    best_params = unflatten_any(best_params)
    best_seed = hpo_result["fit_model_return"]["study"].best_trial.user_attrs["result"]["seed_model"]

    experiment = ClassificationCoClusteringExperiment(
        agent_i=ai,
        # dataset
        n_samples=n_samples,
        n_informative=n_informative,
        n_random=n_random,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        class_sep=class_sep,
        seed_dataset=seed_dataset,
        shift=shift,
        scale=scale,
        n_agents=n_agents,
        p_overlap=p_overlap,
        sequential_split=sequential_split,
        features_groups=features_groups,
        # model
        model=model,
        model_params=best_params,
        seed_model=best_seed,
        # experiment
        raise_on_error=True,
        verbose=1,
    )
    result = experiment.run(return_results=True)[0]
    features = result["load_data_return"]["X"]
    y_true = result["load_data_return"]["y"]
    labels = result["fit_model_return"]["y_pred"]
    unique_labels = np.unique(labels)
    tsne = TSNE(n_components=2, random_state=seed_dataset)
    X_embedded = tsne.fit_transform(features)
    with mpl.rc_context(
        rc={
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times", "serif"],  # Reliable serif fonts
            "mathtext.fontset": "dejavuserif",  # Use DejaVu for math text to avoid missing glyphs
            "axes.unicode_minus": False,  # Use ASCII minus instead of Unicode minus
            "font.size": 12,
            "axes.linewidth": 1.2,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.shadow": False,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "black",
            "grid.alpha": 0.5,
            "axes.grid": True,
            "grid.linewidth": 0.5,
        }
    ):
        cm = 1 / 2.54  # centimeters to inches
        fig_scale = 3.0
        fig, ax = plt.subplots(figsize=(12 * fig_scale * cm, 7 * fig_scale * cm))
        # Create discrete colormap with only the colors you need
        n_clusters = len(np.unique(labels))
        colors = plt.colormaps["tab10"].colors
        markers = ["o", "s", "^", "v", "D", "P", "*", "X", "p", "h"]  # circle, square, triangle, etc.
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                X_embedded[mask, 0],
                X_embedded[mask, 1],
                c=[colors[i % len(colors)]],
                marker=markers[i % len(markers)],
                s=90,
                label=f"Cluster {label}",
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_ylabel("t-SNE feature 2")
        ax.set_xlabel("t-SNE feature 1")
        fig.savefig(
            results_dir
            / f"hypercube_{model}_agent-{ai}_n-samples-{n_samples}_n-informative-{n_informative}_n-random-{n_random}_n-classes-{n_classes}_class-sep-{class_sep}_shift-{shift}_scale-{scale}_n-agents-{n_agents}_p-overlap-{p_overlap}_sequential-split-{sequential_split}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
