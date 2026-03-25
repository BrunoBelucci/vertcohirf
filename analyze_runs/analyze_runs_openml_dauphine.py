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
from sqlalchemy import create_engine, text
import pandas as pd
from ml_experiments.analyze import get_df_runs_from_mlflow_sql
from pathlib import Path
import os
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# %% [markdown]
# # Save Results

# %% [markdown]
# ## Load mlflow runs

# %%
results_dir = Path.cwd().parent / "results" / "real" / "dauphine"
os.makedirs(results_dir, exist_ok=True)

# %%
db_port = 6001
db_name = "cocohirf"
url = f"postgresql://belucci@localhost:{db_port}/{db_name}"
engine = create_engine(url)
query = "SELECT experiments.name from experiments"
experiment_names = pd.read_sql(query, engine)["name"].tolist()

# %%
experiment_names

# %%
experiments_names = [exp for exp in experiment_names if exp.startswith("real-")]

# %%
experiments_names

# %%
query = "SELECT DISTINCT(key) FROM params WHERE key LIKE 'best/%%'"
best_params = pd.read_sql(query, engine)["key"].tolist()

# %%
params_columns = [
    "model",
    "model_alias",
    "dataset_id",
    "n_trials",
    "n_trials_1",
    "dataset_name",
    "standardize",
    "hpo_metric",
    "hpo_metric_2",
	"direction",
    "direction_2",
    "hpo_seed",
    "seed_dataset_order",
    "n_agents",
    "agent_i",
] + best_params

# %%
metrics = [
	"rand_score",
	"adjusted_rand",
	"mutual_info",
	"adjusted_mutual_info",
	"normalized_mutual_info",
	"homogeneity",
	"completeness",
	"v_measure",
	"silhouette",
	"calinski_harabasz_score",
	"davies_bouldin_score",
]
best_mean_metrics = [f"best/{metric}_mean" for metric in metrics]
best_std_metrics = [f"best/{metric}_std" for metric in metrics]

# %%
latest_metrics_columns = [
    "fit_model_return_elapsed_time",
    "max_memory_used_after_fit",
    "max_memory_used",
	"best/min_n_clusters",
    "best/elapsed_time",
]
latest_metrics_columns += best_mean_metrics + best_std_metrics

# %%
tags_columns = ["raised_exception", "EXCEPTION", "mlflow.parentRunId", "Last step finished", "hpo_stage"]

# %%
runs_columns = ['run_uuid', 'status', 'start_time', 'end_time']
experiments_columns = []
other_table = 'params'
other_table_keys = params_columns
df_tags = get_df_runs_from_mlflow_sql(engine, runs_columns=runs_columns, experiments_columns=experiments_columns, experiments_names=experiments_names, other_table='tags', other_table_keys=tags_columns)
df_params = get_df_runs_from_mlflow_sql(engine, runs_columns=['run_uuid'], experiments_columns=experiments_columns, experiments_names=experiments_names, other_table=other_table, other_table_keys=other_table_keys)
df_latest_metrics = get_df_runs_from_mlflow_sql(engine, runs_columns=['run_uuid'], experiments_columns=experiments_columns, experiments_names=experiments_names, other_table='latest_metrics', other_table_keys=latest_metrics_columns)


# %%
dataset_characteristics = pd.read_csv(results_dir / "datasets_characteristics.csv", index_col=0)
dataset_characteristics.index = dataset_characteristics["openml_id"].astype(str)

# %%
df_runs_raw = df_tags.join(df_latest_metrics)
df_runs_raw = df_runs_raw.join(df_params)
df_runs_raw = df_runs_raw.join(dataset_characteristics, on='dataset_id', rsuffix='_dataset')
df_runs_raw.to_csv(results_dir / 'df_runs_raw.csv', index=True)

# %%
df_runs_raw = pd.read_csv(results_dir / "df_runs_raw.csv", index_col=0)
df_runs_raw.loc[df_runs_raw["model"].isna(), "model"] = df_runs_raw.loc[df_runs_raw["model"].isna(), "model_alias"]
df_runs_raw.loc[df_runs_raw["n_trials"].isna(), "n_trials"] = df_runs_raw.loc[df_runs_raw["n_trials"].isna(), "n_trials_1"]
# df_runs_raw["n_trials"] = df_runs_raw["n_trials"].astype(int)
df_runs_raw.loc[df_runs_raw["hpo_metric"].isna(), "hpo_metric"] = df_runs_raw.loc[df_runs_raw["hpo_metric"].isna(), "hpo_metric_2"]
df_runs_raw.loc[df_runs_raw["direction"].isna(), "direction"] = df_runs_raw.loc[df_runs_raw["direction"].isna(), "direction_2"]
# mask = df_runs_raw["model"].str.contains("CoHiRF")
# df_runs_raw.loc[mask, "model"] = df_runs_raw.loc[mask].apply(lambda row: f"{row['model']}-{row['n_trials']}", axis=1)
df_runs_raw_parents = df_runs_raw.copy()
df_runs_raw_parents = df_runs_raw_parents.loc[df_runs_raw_parents["mlflow.parentRunId"].isna()]
df_runs_raw_parents["n_trials"] = df_runs_raw_parents["n_trials"].astype(int)
df_runs_raw_parents["model"] = df_runs_raw_parents["model"] + "-" + df_runs_raw_parents["n_trials"].astype(str)
df_runs_raw_parents["dataset_id"] = df_runs_raw_parents["dataset_id"].astype(int)
df_runs_raw_parents["hpo_seed"] = df_runs_raw_parents["hpo_seed"].astype(int)
df_runs_raw_parents["seed_dataset_order"] = df_runs_raw_parents["seed_dataset_order"].astype(int)
df_runs_raw_parents["n_agents"] = df_runs_raw_parents["n_agents"].astype(int)
df_runs_raw_parents["agent_i"] = df_runs_raw_parents["agent_i"].astype(str)

# %%
df_runs_raw_parents.head(5)

# %%
df_runs_raw_parents[["model", "dataset_id", "max_memory_used_after_fit"]].sort_values(by="max_memory_used_after_fit", ascending=False)

# %% [markdown]
# # Count children runs that failed

# %%
# Create a mapping of run_id to parent_id
parent_map = df_runs_raw.set_index(df_runs_raw.index)["mlflow.parentRunId"].to_dict()

# Function to find the root parent
def find_root_parent(run_id, parent_map, max_depth=100):
    """Find the root parent by following the parent chain"""
    current_id = run_id
    depth = 0

    while current_id in parent_map and pd.notna(parent_map[current_id]) and depth < max_depth:
        current_id = parent_map[current_id]
        depth += 1

    return current_id

runs_failed = df_runs_raw.copy()
# get only children runs
runs_failed = runs_failed.loc[~runs_failed["mlflow.parentRunId"].isna()]
# get the top parent run for each child
runs_failed["root_parent_id"] = runs_failed.index.map(lambda x: find_root_parent(x, parent_map))
# filter only failed runs
runs_failed = runs_failed.loc[(runs_failed["raised_exception"] == True) | (runs_failed["status"] != "FINISHED")]
# count how many children runs failed for each parent
failed_counts = runs_failed.groupby("root_parent_id").size().reset_index(name="failed_children_count")
failed_counts

# %% [markdown]
# ## Delete duplicate runs (if any) and complete some models that cannot run with some datasets

# %%
non_duplicate_columns = [
    "model",
    "dataset_id",
	"standardize",
	"hpo_metric",
	"hpo_seed",
	"seed_dataset_order",
	"n_agents",
	"agent_i",
]
# df_runs_parents.loc[df_runs_parents["best/n_clusters_"]*0.5 > df_runs_parents["n_instances"], "best/adjusted_rand"] = 
df_runs_parents = df_runs_raw_parents.dropna(axis=0, how="all", subset=["best/adjusted_rand_mean"]).copy()
# add back runs that were not evaluated because we judged too many clusters (but they run anyway)
# df_valid_runs = df_runs_raw_parents.loc[df_runs_raw_parents["best/n_clusters_"] > df_runs_raw_parents["n_instances"]*0.5].copy()
# df_runs_parents = pd.concat([df_runs_parents, df_valid_runs], axis=0)
df_runs_parents = df_runs_parents.loc[(~df_runs_parents.duplicated(non_duplicate_columns))]
# fill missing values with "None"
df_runs_parents = df_runs_parents.fillna("None")

# %%
df_to_cat = []
hpo_metrics = [
    "adjusted_rand_mean",
]
standardize = [True]
hpo_seed = [i for i in range(5)]
fill_value = pd.NA
fill_columns = [
    "best/adjusted_rand_mean",
    # "best/adjusted_mutual_info",
    # "best/calinski_harabasz_score",
    # "best/silhouette",
    # "best/davies_bouldin_score",
    # "best/normalized_mutual_info",
]
n_agents = [2]
agents_i = [0]

# %%
# # Too memory intensive
# dataset_ids_to_complete = [182, 1478, 1568]
# model_names = ["VeCoHiRF-SC-SRGF-50"]
# for dataset_id in dataset_ids_to_complete:
#     for agent_i in agents_i:
#         for n_agent in n_agents:
#             for model_name in model_names:
#                 for hpo_metric in hpo_metrics:
#                     for std in standardize:
#                         for seed in hpo_seed:
#                             new_row = {
#                                 "dataset_id": dataset_id,
#                                 "model": model_name,
#                                 "hpo_metric": hpo_metric,
#                                 "standardize": std,
#                                 "hpo_seed": seed,
#                                 "n_agents": n_agent,
#                                 "agent_i": agent_i,
#                             }
#                             for col in fill_columns:
#                                 new_row[col] = fill_value
#                             df_to_cat.append(new_row)

# %%
# df_runs_parents = pd.concat([df_runs_parents, pd.DataFrame(df_to_cat)], axis=0)

# %% [markdown]
# # Missing

# %%
model_nickname = df_runs_parents['model'].unique().tolist()
model_nickname.sort()
model_nickname

# %%
models_names = [
    # collaborative methods to be compared with
    "CoresetKMeans-50",
    "DistributedKMeans-50",
    "V2way-50",
    "VPC-50",
    # base methods
    # "KMeans",
    # "KernelRBFKMeans",
    # "DBSCAN",
    # "SpectralSubspaceRandomization",
    # VeCoHiRF methods
    "VeCoHiRF-1iter-50",
    "VeCoHiRF-top-down-1iter-50",
    "VeCoHiRF-DBSCAN-1iter-50",
    "VeCoHiRF-DBSCAN-top-down-1iter-50",
    "VeCoHiRF-KernelRBF-1iter-50",
    "VeCoHiRF-KernelRBF-top-down-1iter-50",
    "VeCoHiRF-SC-SRGF-1R-1iter-50",
    # "VeCoHiRF-SC-SRGF-1R-top-down-1iter",
]
datasets_ids = [46773, 46778, 46779]
ns_agents = [2, 3, 4, 5, 6]
hpo_seed = [i for i in range(5)]
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]
agent_i = "nan"
standardize = True
combination_df = []
for model in models_names:
    for dataset_id in datasets_ids:
        for n_agents in ns_agents:
            for hpo_metric in hpo_metrics:
                for seed in hpo_seed:
                    combination_df.append(
                        dict(
                            model=model,
                            hpo_metric=hpo_metric,
                            direction="maximize",
                            dataset_id=dataset_id,
                            hpo_seed=seed,
                            seed_dataset_order=seed,
                            standardize=standardize,
                            n_agents=n_agents,
                            agent_i=agent_i,
                        )
                    )
combination_df = pd.DataFrame(combination_df)

unique_columns = [
    "model",
    "dataset_id",
    "standardize",
    "hpo_metric",
    "direction",
    "hpo_seed",
    "seed_dataset_order",
    "n_agents",
    "agent_i",
]
# compare combinations_df with df_runs_parents to get missing combinations
df_missing = combination_df.merge(
    df_runs_parents,
    on=unique_columns,
    how="left",
    indicator=True,
)
df_missing = df_missing.loc[df_missing["_merge"] == "left_only"][unique_columns]
df_missing

# %%
models_names = [
    "CoresetKMeans-50",
    "DistributedKMeans-50",
    "V2way-50",
    "VPC-50",
    "VeCoHiRF-1iter-50",
    "VeCoHiRF-top-down-1iter-50",
    "VeCoHiRF-DBSCAN-1iter-50",
    "VeCoHiRF-DBSCAN-top-down-1iter-50",
    "VeCoHiRF-KernelRBF-1iter-50",
    "VeCoHiRF-KernelRBF-top-down-1iter-50",
    # "VeCoHiRF-SC-SRGF-1R-1iter-50",
]
datasets_ids = [46779, 46783, 40685, 46778, 46782, 182, 46773, 46776, 1568, 554]
n_agents = 3
hpo_seed = [i for i in range(5)]
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]
agent_i = 'nan'
standardize = True
combination_df = []
for model in models_names:
    for dataset_id in datasets_ids:
        for hpo_metric in hpo_metrics:
            for seed in hpo_seed:
                combination_df.append(
                    dict(
                        model=model,
                        hpo_metric=hpo_metric,
                        direction="maximize",
                        dataset_id=dataset_id,
                        hpo_seed=seed,
                        seed_dataset_order=seed,
                        standardize=standardize,
                        n_agents=n_agents,
                        agent_i=agent_i,
                    )
                )
combination_df = pd.DataFrame(combination_df)

unique_columns = [
    "model",
    "dataset_id",
    "standardize",
    "hpo_metric",
    "direction",
    "hpo_seed",
    "seed_dataset_order",
    "n_agents",
    "agent_i",
]
# compare combinations_df with df_runs_parents to get missing combinations
df_missing = combination_df.merge(
    df_runs_parents,
    on=unique_columns,
    how="left",
    indicator=True,
)
df_missing = df_missing.loc[df_missing["_merge"] == "left_only"][unique_columns]
df_missing

# %%
models_names = [
    # collaborative methods to be compared with
    # "CoresetKMeans",
    # "DistributedKMeans",
    # "V2way",
    # "VPC",
    # base methods
    "KMeans-50",
    "KernelRBFKMeans-50",
    "DBSCAN-50",
    # "SpectralSubspaceRandomization-50",
    # VeCoHiRF methods
    # "VeCoHiRF-1iter",
    # "VeCoHiRF-top-down-1iter",
    # "VeCoHiRF-DBSCAN-1iter",
    # "VeCoHiRF-DBSCAN-top-down-1iter",
    # "VeCoHiRF-KernelRBF-1iter",
    # "VeCoHiRF-KernelRBF-top-down-1iter",
    # "VeCoHiRF-SC-SRGF-1R-1iter",
    # "VeCoHiRF-SC-SRGF-1R-top-down-1iter",
]
datasets_ids = [46779, 46783, 40685, 46778, 46782, 182, 46773, 46776, 1568, 554]
n_agents = 3
agent_is = [i for i in range(n_agents)]
hpo_seed = [i for i in range(5)]
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]
agent_i = None
standardize = True
combination_df = []
for model in models_names:
    for dataset_id in datasets_ids:
        for agent_i in agent_is:
            for hpo_metric in hpo_metrics:
                for seed in hpo_seed:
                    combination_df.append(
                        dict(
                            model=model,
                            hpo_metric=hpo_metric,
                            direction="maximize",
                            dataset_id=dataset_id,
                            hpo_seed=seed,
                            seed_dataset_order=seed,
                            standardize=standardize,
                            n_agents=n_agents,
                            agent_i=agent_i,
                        )
                    )
combination_df = pd.DataFrame(combination_df)
combination_df["agent_i"] = combination_df["agent_i"].astype(float).astype(str)

unique_columns = [
    "model",
    "dataset_id",
    "standardize",
    "hpo_metric",
    "direction",
    "hpo_seed",
    "seed_dataset_order",
    "n_agents",
    "agent_i",
]
# compare combinations_df with df_runs_parents to get missing combinations
df_missing = combination_df.merge(
    df_runs_parents,
    on=unique_columns,
    how="left",
    indicator=True,
)
df_missing = df_missing.loc[df_missing["_merge"] == "left_only"][unique_columns]
df_missing

# %%
unique_columns = [
    "model",
    "dataset_id",
    "standardize",
    "hpo_metric",
    "direction",
	"hpo_seed",
    "seed_dataset_order",
    "n_agents",
    "agent_i",
]
# compare combinations_df with df_runs_parents to get missing combinations
df_missing = combination_df.merge(
    df_runs_parents,
    on=unique_columns,
    how="left",
    indicator=True,
)
df_missing = df_missing.loc[df_missing["_merge"] == "left_only"][unique_columns]
df_missing

# %%
# Join df_runs_raw_parents into df_missing using non_duplicate_columns to get the EXCEPTION column
df_missing_with_exception = df_missing.merge(
    df_runs_raw_parents[unique_columns + ["raised_exception", "EXCEPTION", "Last step finished"]],
    how="left",
    left_on=unique_columns,
    right_on=unique_columns,
)
df_missing_with_exception[unique_columns + ["raised_exception", "EXCEPTION", "Last step finished"]].sort_values(
    by="EXCEPTION"
)[["model", "dataset_id", "EXCEPTION"]]

# %%
# get rid of -50
df_missing["model"] = df_missing["model"].str.replace("-50", "")

# %%
df_missing

# %%
df_missing.to_csv(results_dir / "df_missing.csv", index=False)

# %%
models_names = [
    "CoresetKMeans-50",
    "DistributedKMeans-50",
    "V2way-50",
    "VPC-50",
    "VeCoHiRF-1iter-50",
    "VeCoHiRF-top-down-1iter-50",
    "VeCoHiRF-DBSCAN-1iter-50",
    "VeCoHiRF-DBSCAN-top-down-1iter-50",
    "VeCoHiRF-KernelRBF-1iter-50",
    "VeCoHiRF-KernelRBF-top-down-1iter-50",
    # "VeCoHiRF-SC-SRGF-1R-1iter-50",
]
datasets_ids = [46773, 46778, 46783, 46779, 1568]
df = df_runs_parents.copy()
df = df.loc[df["model"].isin(models_names) & df["dataset_id"].isin(datasets_ids)]
df[["model", "dataset_id", "max_memory_used", "fit_model_return_elapsed_time"]].sort_values(
    by="max_memory_used", ascending=False
)


# %%
df_high_mem_high_time = (
    df.loc[(df["max_memory_used"] > 10000) | (df["fit_model_return_elapsed_time"] > 3600), ["model", "dataset_id"]] 
    .drop_duplicates()
    .sort_values(["model", "dataset_id"])
    .reset_index(drop=True)
)
df_high_mem_high_time

# %%
# drop "-50" from model names
df_high_mem_high_time["model"] = df_high_mem_high_time["model"].str.replace("-50", "")
df_high_mem_high_time.to_csv("/home/belucci/code/vertcohirf/run_scripts/high_mem_tim_runs.csv", index=False)

# %% [markdown]
# # Plot n agets

# %%
df = df_runs_parents.copy()
models_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DistributedKMeans-50": "DistributedKMeans",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-top-down-1iter-50": "R-VertCoHiRF",
    # "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    # "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    # "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    # "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
}
dataset_id = 46778
# datasets_ids = [46773, 46778, 46779]
df = df.loc[df["dataset_id"] == dataset_id]
df = df.loc[df["model"].isin(models_names.keys())]
df = df.replace({"model": models_names})
# plot n_agents vs best/metric for each model using seaborn
metrics = {
    "adjusted_rand_mean": "Adjusted Rand Index",
    "silhouette_mean": "Silhouette Score",
    "calinski_harabasz_score_mean": "Calinski-Harabasz Score",
}
for metric in metrics.keys():
    hpo_metric = metric
    df_metric = df.loc[df["hpo_metric"] == hpo_metric]
    with mpl.rc_context(
        rc={
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times", "serif"],  # Reliable serif fonts
            "mathtext.fontset": "dejavuserif",  # Use DejaVu for math text to avoid missing glyphs
            "axes.unicode_minus": False,  # Use ASCII minus instead of Unicode minus
            "font.size": 12,
            "axes.linewidth": 1.2,
            "axes.labelsize": 32,
            "axes.titlesize": 16,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
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
        sns.lineplot(
            data=df_metric,
            x="n_agents",
            y=f"best/{hpo_metric}",
            hue="model",
            errorbar="ci",
            ax=ax,
            hue_order=models_names.values(),
            style="model",
            markers=True,
            markersize=12,
        )
        # move legend outside the plot
        # hide legend title
        plt.legend().set_title("")
        plt.legend(loc='lower right')
        # set x ticks to be integers
        plt.xticks([2, 3, 4, 5])
        # y > 0
        plt.ylim(bottom=0)
        plt.xlabel("Number of Agents")
        plt.ylabel(f"{metrics[hpo_metric]}")
        plt.show()
        fig.savefig(results_dir / f"n_agents_vs_best_{hpo_metric}_dataset_{dataset_id}.pdf", bbox_inches='tight')

# %% [markdown]
# # Katia

# %%
df = df_runs_parents.copy()
models_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DistributedKMeans-50": "DistributedKMeans",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    "VeCoHiRF-1iter-50": "VertCoHiRF",
    # "VeCoHiRF-top-down-1iter-50": "R-VertCoHiRF",
    # "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    # "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    # "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    # "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
}
dataset_id = 46779
# datasets_ids = [46773, 46778, 46779]
df = df.loc[df["dataset_id"] == dataset_id]
df = df.loc[df["model"].isin(models_names.keys())]
df = df.replace({"model": models_names})
# plot n_agents vs best/metric for each model using seaborn
metrics = {
    "adjusted_rand_mean": "Adjusted Rand Index",
    # "silhouette_mean": "Silhouette Score",
    "calinski_harabasz_score_mean": "Calinski-Harabasz Score",
}
for metric in metrics.keys():
    hpo_metric = metric
    dataset_name = df["dataset_name"].iloc[0]
    df_metric = df.loc[df["hpo_metric"] == hpo_metric]
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
        sns.lineplot(
            data=df_metric,
            x="n_agents",
            y=f"best/{hpo_metric}",
            hue="model",
            errorbar="ci",
            ax=ax,
            hue_order=models_names.values(),
            style="model",
            markers=True,
            markersize=12,
        )
        # move legend outside the plot
        # hide legend title
        plt.legend().set_title("")
        plt.legend(loc="lower right")
        # set x ticks to be integers
        plt.xticks([2, 3, 4, 5])
        # y > 0
        plt.ylim(bottom=0)
        plt.xlabel("Number of Agents")
        plt.ylabel(f"{metrics[hpo_metric]}")
        plt.show()
        fig.savefig(
            results_dir / f"n_agents_vs_best_{hpo_metric}_dataset_{dataset_name}_katia.pdf", bbox_inches="tight"
        )

# %%
df = df_runs_parents.copy()
models_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DistributedKMeans-50": "DistributedKMeans",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    # "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-top-down-1iter-50": "VertCoHiRF",
    # "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    # "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    # "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    # "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
}
dataset_id = 46779
# datasets_ids = [46773, 46778, 46779]
df = df.loc[df["dataset_id"] == dataset_id]
df = df.loc[df["model"].isin(models_names.keys())]
df = df.replace({"model": models_names})
# plot n_agents vs best/metric for each model using seaborn
metrics = {
    # "adjusted_rand_mean": "Adjusted Rand Index",
    "silhouette_mean": "Silhouette Score",
    # "calinski_harabasz_score_mean": "Calinski-Harabasz Score",
}
for metric in metrics.keys():
    hpo_metric = metric
    dataset_name = df["dataset_name"].iloc[0]
    df_metric = df.loc[df["hpo_metric"] == hpo_metric]
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
        sns.lineplot(
            data=df_metric,
            x="n_agents",
            y=f"best/{hpo_metric}",
            hue="model",
            errorbar="ci",
            ax=ax,
            hue_order=models_names.values(),
            style="model",
            markers=True,
            markersize=12,
        )
        # move legend outside the plot
        # hide legend title
        plt.legend().set_title("")
        plt.legend(loc="lower right")
        # set x ticks to be integers
        plt.xticks([2, 3, 4, 5])
        # y > 0
        plt.ylim(bottom=0)
        plt.xlabel("Number of Agents")
        plt.ylabel(f"{metrics[hpo_metric]}")
        plt.show()
        fig.savefig(
            results_dir / f"n_agents_vs_best_{hpo_metric}_dataset_{dataset_name}_katia.pdf", bbox_inches="tight"
        )

# %%
df = df_runs_parents.copy()
models_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DistributedKMeans-50": "DistributedKMeans",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    # "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-top-down-1iter-50": "VertCoHiRF",
    # "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    # "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    # "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    # "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
}
dataset_id = 46773
# datasets_ids = [46773, 46778, 46779]
df = df.loc[df["dataset_id"] == dataset_id]
df = df.loc[df["model"].isin(models_names.keys())]
df = df.replace({"model": models_names})
# plot n_agents vs best/metric for each model using seaborn
metrics = {
    "adjusted_rand_mean": "Adjusted Rand Index",
    "silhouette_mean": "Silhouette Score",
    "calinski_harabasz_score_mean": "Calinski-Harabasz Score",
}
for metric in metrics.keys():
    hpo_metric = metric
    dataset_name = df["dataset_name"].iloc[0]
    df_metric = df.loc[df["hpo_metric"] == hpo_metric]
    with mpl.rc_context(
        rc={
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times", "serif"],  # Reliable serif fonts
            "mathtext.fontset": "dejavuserif",  # Use DejaVu for math text to avoid missing glyphs
            "axes.unicode_minus": False,  # Use ASCII minus instead of Unicode minus
            "font.size": 12,
            "axes.linewidth": 1.2,
            "axes.labelsize": 32,
            "axes.titlesize": 32,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
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
        sns.lineplot(
            data=df_metric,
            x="n_agents",
            y=f"best/{hpo_metric}",
            hue="model",
            errorbar="ci",
            ax=ax,
            hue_order=models_names.values(),
            style="model",
            markers=True,
            markersize=12,
        )
        # move legend outside the plot
        # hide legend title
        plt.legend().set_title("")
        plt.legend(loc="lower right")
        # set x ticks to be integers
        plt.xticks([2, 3, 4, 5])
        # y > 0
        plt.ylim(bottom=0)
        plt.xlabel("Number of Agents")
        plt.ylabel(f"{metrics[hpo_metric]}")
        plt.show()
        fig.savefig(results_dir / f"n_agents_vs_best_{hpo_metric}_dataset_{dataset_name}_katia.pdf", bbox_inches="tight")

# %%
df = df_runs_parents.copy()
models_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DistributedKMeans-50": "DistributedKMeans",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    "VeCoHiRF-1iter-50": "VertCoHiRF",
    # "VeCoHiRF-top-down-1iter-50": "VertCoHiRF",
    # "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    # "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    # "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    # "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
}
dataset_id = 46778
# datasets_ids = [46773, 46778, 46779]
df = df.loc[df["dataset_id"] == dataset_id]
df = df.loc[df["model"].isin(models_names.keys())]
df = df.replace({"model": models_names})
# plot n_agents vs best/metric for each model using seaborn
metrics = {
    "adjusted_rand_mean": "Adjusted Rand Index",
    # "silhouette_mean": "Silhouette Score",
    "calinski_harabasz_score_mean": "Calinski-Harabasz Score",
}
for metric in metrics.keys():
    hpo_metric = metric
    dataset_name = df["dataset_name"].iloc[0]
    df_metric = df.loc[df["hpo_metric"] == hpo_metric]
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
        sns.lineplot(
            data=df_metric,
            x="n_agents",
            y=f"best/{hpo_metric}",
            hue="model",
            errorbar="ci",
            ax=ax,
            hue_order=models_names.values(),
            style="model",
            markers=True,
            markersize=12,
        )
        # move legend outside the plot
        # hide legend title
        plt.legend().set_title("")
        plt.legend(loc="lower right")
        # set x ticks to be integers
        plt.xticks([2, 3, 4, 5])
        # y > 0
        plt.ylim(bottom=0)
        plt.xlabel("Number of Agents")
        plt.ylabel(f"{metrics[hpo_metric]}")
        plt.show()
        fig.savefig(
            results_dir / f"n_agents_vs_best_{hpo_metric}_dataset_{dataset_name}_katia.pdf", bbox_inches="tight"
        )

# %%
df = df_runs_parents.copy()
models_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DistributedKMeans-50": "DistributedKMeans",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    # "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-top-down-1iter-50": "VertCoHiRF",
    # "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    # "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    # "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    # "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
}
dataset_id = 46778
# datasets_ids = [46773, 46778, 46779]
df = df.loc[df["dataset_id"] == dataset_id]
df = df.loc[df["model"].isin(models_names.keys())]
df = df.replace({"model": models_names})
# plot n_agents vs best/metric for each model using seaborn
metrics = {
    # "adjusted_rand_mean": "Adjusted Rand Index",
    "silhouette_mean": "Silhouette Score",
    # "calinski_harabasz_score_mean": "Calinski-Harabasz Score",
}
for metric in metrics.keys():
    hpo_metric = metric
    dataset_name = df["dataset_name"].iloc[0]
    df_metric = df.loc[df["hpo_metric"] == hpo_metric]
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
        sns.lineplot(
            data=df_metric,
            x="n_agents",
            y=f"best/{hpo_metric}",
            hue="model",
            errorbar="ci",
            ax=ax,
            hue_order=models_names.values(),
            style="model",
            markers=True,
            markersize=12,
        )
        # move legend outside the plot
        # hide legend title
        plt.legend().set_title("")
        plt.legend(loc="lower right")
        # set x ticks to be integers
        plt.xticks([2, 3, 4, 5])
        # y > 0
        plt.ylim(bottom=0)
        plt.xlabel("Number of Agents")
        plt.ylabel(f"{metrics[hpo_metric]}")
        plt.show()
        fig.savefig(
            results_dir / f"n_agents_vs_best_{hpo_metric}_dataset_{dataset_name}_katia.pdf", bbox_inches="tight"
        )


# %% [markdown]
# # Tables

# %%
def get_parameters_string(row):
    parameter_names = {
		"best/alpha": "\\alpha",
		"best/avg_dims": "d",
		"best/base_model_kwargs/eps": "\\epsilon",
		"best/base_model_kwargs/min_samples": "n_{\\text{min}}",
		"best/base_model_kwargs/n_clusters": "C",
		"best/c": "c",
		"best/cohirf_kwargs/base_model_kwargs/eps": "\\epsilon",
		"best/cohirf_kwargs/base_model_kwargs/min_samples": "n_{\\text{min}}",
		"best/cohirf_kwargs/kmeans_n_clusters": "C",
		"best/cohirf_kwargs/n_features": "q",
		"best/cohirf_kwargs/repetitions": "R",
		"best/damping": "\\lambda",
		# "best/density_threshold": "\\tau",
		"best/eps": "\\epsilon",
		"best/kmeans_n_clusters": "C",
		"best/lambda_": "\\lambda",
		"best/min_bin_freq": "bin_{\\text{min}}",
		"best/min_cluster_size": "C_{\\text{min}}",
		"best/min_samples": "n_{\\text{min}}",
		"best/n_clusters": "C",
		"best/n_features": "q",
		# "best/n_partitions": "P",
		"best/n_similarities": "m",
		"best/p": "p",
		"best/repetitions": "R",
		"best/sampling_ratio": "r",
		"best/sc_n_clusters": "C",
		"best/transform_kwargs/gamma": "\\gamma",
	}
    first = True
    str = ""
    for p in parameter_names.keys():
        if not pd.isna(row[p]) and row[p] != "None":
            if not first:
                str += "; "
            else:
                first = False
            value = float(row[p])
            if value.is_integer():
                value = int(value)
                str += f"${parameter_names[p]}={value}$"
            else:
                str += f"${parameter_names[p]}={value:0.2f}$"
    return str


# %%
def highlight_max(df, column_name, level=0):
    df_column = df[column_name]
    if level is None:
        max_values = pd.Series([df_column.max()] * len(df_column), index=df_column.index)
    else:
        max_values = df_column.groupby(level=level).transform('max')
    is_highlighted = df_column.round(3) == max_values.round(3)
    df_css = df.copy().astype(str)
    df_css.loc[:, :] = ''
    df_css[is_highlighted] = 'font-weight: bold'
    return df_css


# %%
def highlight_min(df, column_name, level=0):
    df_column = df[column_name]
    min_values = df_column.groupby(level=level).transform("min")
    is_highlighted = df_column.round(3) == min_values.round(3)
    df_css = df.copy().astype(str)
    df_css.loc[:, :] = ""
    df_css[is_highlighted] = "font-weight: bold"
    return df_css


# %%
def highlight_max_index(series_index, df_column, level=0):
    max_values = df_column.groupby(level=level).transform('max')
    is_highlighted = df_column.round(3) == max_values.round(3)
    series_css = series_index.copy().astype(str)
    series_css[:] = ''
    series_css[is_highlighted.values] = 'font-weight: bold'
    return series_css


# %%
def underline_2nd_max(df, column_name, level=0):
    df_column = df[column_name]
    # get the second max value
    if level is None:
        second_max_values = pd.Series([df_column.round(3).drop_duplicates().nlargest(2).iloc[-1]] * len(df_column), index=df_column.index)
    else:
        second_max_values = df_column.groupby(level=level).transform(lambda x: x.round(3).drop_duplicates().nlargest(2).iloc[-1])
    is_underlined = df_column.round(3) == second_max_values.round(3)
    df_css = df.copy().astype(str)
    df_css.loc[:, :] = ''
    df_css[is_underlined] = 'underline: --latex--rwrap'
    return df_css


# %%
def underline_2nd_min(df, column_name, level=0):
    df_column = df[column_name]
    # get the second min value
    second_min_values = df_column.groupby(level=level).transform(
        lambda x: x.round(3).drop_duplicates().nsmallest(2).iloc[-1]
    )
    is_underlined = df_column.round(3) == second_min_values.round(3)
    df_css = df.copy().astype(str)
    df_css.loc[:, :] = ""
    df_css[is_underlined] = "underline: --latex--rwrap"
    return df_css


# %%
def underline_2nd_max_index(series_index, df_column, level=0):
    # get the second max value
    second_max_values = df_column.groupby(level=level).transform(lambda x: x.nlargest(2).iloc[-1])
    is_underlined = df_column.round(3) == second_max_values.round(3)
    series_css = series_index.copy().astype(str)
    series_css.loc[:] = ''
    series_css[is_underlined.values] = 'underline: --latex--rwrap'
    return series_css


# %% [markdown]
# # Compared to base model

# %%
def get_metrics(df, model_groups, hpo_metrics, metrics_rename, agg_mode="mean", improv="pct", raw_metrics=False):
    df["model_group"] = df["model"].apply(
        lambda x: next((group for group, models in model_groups.items() if x in models), "Other")
    )
    join_columns = ["dataset_name", "hpo_seed", "hpo_metric", "model_group"]
    df_base_models = df.loc[df["model"].isin(model_groups.keys())].copy()
    df_not_base_models = df.loc[~df["model"].isin(model_groups.keys())].copy()
    df_not_base_models = df_not_base_models.set_index(join_columns)
    df_not_base_models.drop(columns=["agent_i"], inplace=True)
    # each row in df_base_models is "exploded" into multiple rows for each collaborative model
    df_collab = df_base_models.join(df_not_base_models, on=join_columns, rsuffix="_collab")
    # for each metric, compute the improvement of each collaborative model over the base model
    if improv == "pct":
        for hpo_metric in hpo_metrics:
            df_collab[f"{improv}_improvement/{hpo_metric}"] = (
                (df_collab[f"best/{hpo_metric}_collab"] - df_collab[f"best/{hpo_metric}"])
                / df_collab[f"best/{hpo_metric}"]
                * 100
            )
    elif improv == "diff":
        for hpo_metric in hpo_metrics:
            df_collab[f"{improv}_improvement/{hpo_metric}"] = (
                df_collab[f"best/{hpo_metric}_collab"] - df_collab[f"best/{hpo_metric}"]
            )
    else:
        raise ValueError(f"Unknown improv method: {improv}")

    dfs_metrics = {}

    for hpo_metric, metric_rename in zip(hpo_metrics, metrics_rename):
        columns_to_keep = ["dataset_name", "model_collab", "hpo_seed", "agent_i", f"{improv}_improvement/{hpo_metric}"]
        if raw_metrics:
            columns_to_keep += [f"best/{hpo_metric}_collab", f"best/{hpo_metric}"]
        df_metric = df_collab.loc[df_collab["hpo_metric"] == hpo_metric][columns_to_keep].rename(
            columns={f"{improv}_improvement/{hpo_metric}": metric_rename}
        )
        df_metric = df_metric.dropna(subset=[metric_rename])
        df_metric = df_metric.set_index(["dataset_name", "model_collab", "hpo_seed", "agent_i"])
        df_metric = df_metric.astype({metric_rename: float})
        dfs_metrics[metric_rename] = df_metric

    df_metrics = pd.concat(dfs_metrics.values(), axis=1, join="outer")
    df_metrics = df_metrics.reset_index()

    if agg_mode is not None:
        # aggregate for each dataset_name, model_collab and hpo_seed then do the mean and std across hpo_seeds

        # drop columns where we are going to aggregate
        df_metrics = df_metrics.drop(columns=["agent_i"]) 
        # aggregate
        df_metrics = df_metrics.groupby(["dataset_name", "model_collab", "hpo_seed"]).agg([agg_mode])
        # flatten multiindex columns
        df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
        # reset index
        df_metrics = df_metrics.reset_index()
        # mean across hpo_seed
        df_metrics = df_metrics.drop(columns=["hpo_seed"])
        df_metrics = df_metrics.groupby(["dataset_name", "model_collab"]).agg(["mean", "std"])
        df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
        # Rename index levels
        df_metrics.index.names = ["Dataset", "Model"]

        for metric in metrics_rename:
            df_metrics[f"{metric}"] = (
                df_metrics[f"{metric} {agg_mode} mean"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
                + " $\\pm$ "
                + df_metrics[f"{metric} {agg_mode} std"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
            )

    return df_metrics


# %%
sorted(df_runs_parents['model'].unique())

# %%
model_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DBSCAN-50": "DBSCAN",
    "DistributedKMeans-50": "DistributedKMeans",
    "KMeans-50": "KMeans",
    "KernelRBFKMeans-50": "KernelRBFKMeans",
    "SpectralSubspaceRandomization-50": "SC-SRGF",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
    "VeCoHiRF-top-down-1iter-50": "R-VertCoHiRF",
}

dataset_names = {
    "binary_alpha_digits": "binary-alpha-digits",
	"mnist_784": "mnist",
}  # otherwise we get an error in latex

# Filter runs
df = df_runs_parents.copy()
df = df.loc[df['standardize'] == True]
df = df.loc[df["n_agents"] == 3]
df = df.loc[df['model'].isin(model_names.keys())]
df = df.loc[df["hpo_seed"].isin(range(5))]
df = df.replace({"model": model_names})
df = df.replace({"dataset_name": dataset_names})
df = df[["model", "dataset_name", "agent_i", "hpo_seed", "hpo_metric", "best/adjusted_rand_mean", "best/silhouette_mean", "best/calinski_harabasz_score_mean"]]

# define model groups
model_groups = {
    "KMeans": [
        "KMeans",
        "CoresetKMeans",
        "DistributedKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "R-VertCoHiRF",
    ],
    "KernelKMeans": [
        "KernelRBFKMeans",
        "VertCoHiRF-KernelRBF",
        "R-VertCoHiRF-KernelRBF",
    ],
    "DBSCAN": [
        "DBSCAN",
        "VertCoHiRF-DBSCAN",
        "R-VertCoHiRF-DBSCAN",
    ],
    "SC-SRGF": [
        "SC-SRGF",
        "VertCoHiRF-SC-SRGF",
    ],
}

hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

metrics_rename = [
    "Improv. ARI (\%)",
    "Improv. Silhouette (\%)",
    "Improv. Calinski (\%)",
]

agg_mode = "median"
df_metrics = get_metrics(df, model_groups, hpo_metrics, metrics_rename, agg_mode=agg_mode)

# %%
df_metrics

# %%
df_latex = df_metrics.copy()
df_latex = df_latex.reset_index()
# reapply model groups
df_latex["Base Model"] = df_latex["Model"].apply(
    lambda x: next((group for group, models in model_groups.items() if x in models), "Other")
)
# redefine index with model_group
df_latex = df_latex.set_index(["Dataset", "Base Model", "Model"])
# sort by dataset, model_group, model
df_latex = df_latex.sort_index(level=["Dataset", "Base Model", "Model"])
# raise
# print per dataset
for dataset in df_latex.index.get_level_values("Dataset").unique():
    df_print = df_latex.copy()
    df_print = df_print.loc[dataset]
    columns_to_hide = [col for col in df_latex.columns if col not in (metrics_rename)]
    df_print = df_print.style.hide(columns_to_hide, axis=1)
    for col in metrics_rename:
        highlight_metric = partial(highlight_max, column_name=f"{col} {agg_mode} mean")
        underline_2nd_metric = partial(underline_2nd_max, column_name=f"{col} {agg_mode} mean")
        # if col in ["Davies-Bouldin", "Best Time", "HPO Time"]:
        #     highlight_metric = partial(highlight_min, column_name=f"{col} mean")
        #     underline_2nd_metric = partial(underline_2nd_min, column_name=f"{col} mean")
        (
            df_print.apply(highlight_metric, subset=[col, f"{col} {agg_mode} mean"], axis=None).apply(
                underline_2nd_metric, subset=[col, f"{col} {agg_mode} mean"], axis=None
            )
        )

    latex_output = df_print.to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_print.columns) - len(columns_to_hide)),
        # environment="longtable",
        caption=f"Clustering results on dataset {dataset}",
    )

    # fix header
    columns = df_print.index.names + [col for col in df_print.columns if col not in columns_to_hide]
    header_line = " & ".join(columns) + r" \\"

    # split into lines
    latex_output = latex_output.splitlines()
    # remove 5th and 6th line and replace with header_line
    latex_output = latex_output[:4] + [header_line] + latex_output[6:]
    # remove last cline
    # latex_output = latex_output[:-4] + latex_output[-3:]

    latex_output = "\n".join(latex_output)

    print(latex_output)
    print("\n\n")
    print("\pagebreak")

# %% [markdown]
# # Everyone compared to KMeans

# %%
model_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DBSCAN-50": "DBSCAN",
    "DistributedKMeans-50": "DistributedKMeans",
    "KMeans-50": "KMeans",
    "KernelRBFKMeans-50": "KernelRBFKMeans",
    "SpectralSubspaceRandomization-50": "SC-SRGF",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
    "VeCoHiRF-top-down-1iter-50": "R-VertCoHiRF",
}

dataset_names = {
    "binary_alpha_digits": "binary-alpha-digits",
    "mnist_784": "mnist",
}  # otherwise we get an error in latex

# Filter runs
df = df_runs_parents.copy()
df = df.loc[df["standardize"] == True]
df = df.loc[df["n_agents"] == 3]
df = df.loc[df["model"].isin(model_names.keys())]
df = df.loc[df["hpo_seed"].isin(range(5))]
df = df.replace({"model": model_names})
df = df.replace({"dataset_name": dataset_names})
df = df[
    [
        "model",
        "dataset_name",
        "agent_i",
        "hpo_seed",
        "hpo_metric",
        "best/adjusted_rand_mean",
        "best/silhouette_mean",
        "best/calinski_harabasz_score_mean",
    ]
]

# define model groups
model_groups = {
    "KMeans": [
        "KMeans",
        # "KernelKMeans",
        # "DBSCAN",
        # "SC-SRGF",
        "CoresetKMeans",
        "DistributedKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "R-VertCoHiRF",
        "VertCoHiRF-KernelRBF",
        "R-VertCoHiRF-KernelRBF",
        "VertCoHiRF-DBSCAN",
        "R-VertCoHiRF-DBSCAN",
        "VertCoHiRF-SC-SRGF",
    ],
}
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

metrics_rename = [
    "Improv. ARI (%)",
    "Improv. Silhouette (%)",
    "Improv. Calinski (%)",
]

agg_mode = "median"
df_metrics = get_metrics(df, model_groups, hpo_metrics, metrics_rename, agg_mode=agg_mode)

# %%
df_latex = df_metrics.copy()
df_latex = df_latex.reset_index()
# reapply model groups
df_latex["Base Model"] = df_latex["Model"].apply(
    lambda x: next((group for group, models in model_groups.items() if x in models), "Other")
)
# redefine index with model_group
df_latex = df_latex.set_index(["Dataset", "Base Model", "Model"])
# sort by dataset, model_group, model
df_latex = df_latex.sort_index(level=["Dataset", "Base Model", "Model"])
# raise
# print per dataset
for dataset in df_latex.index.get_level_values("Dataset").unique():
    df_print = df_latex.copy()
    df_print = df_print.loc[dataset]
    columns_to_hide = [col for col in df_latex.columns if col not in (metrics_rename)]
    df_print = df_print.style.hide(columns_to_hide, axis=1)
    for col in metrics_rename:
        highlight_metric = partial(highlight_max, column_name=f"{col} {agg_mode} mean")
        underline_2nd_metric = partial(underline_2nd_max, column_name=f"{col} {agg_mode} mean")
        # if col in ["Davies-Bouldin", "Best Time", "HPO Time"]:
        #     highlight_metric = partial(highlight_min, column_name=f"{col} mean")
        #     underline_2nd_metric = partial(underline_2nd_min, column_name=f"{col} mean")
        (
            df_print.apply(highlight_metric, subset=[col, f"{col} {agg_mode} mean"], axis=None).apply(
                underline_2nd_metric, subset=[col, f"{col} {agg_mode} mean"], axis=None
            )
        )

    latex_output = df_print.to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_print.columns) - len(columns_to_hide)),
        # environment="longtable",
        caption=f"Clustering results on dataset {dataset}",
    )

    # fix header
    columns = df_print.index.names + [col for col in df_print.columns if col not in columns_to_hide]
    header_line = " & ".join(columns) + r" \\"

    # split into lines
    latex_output = latex_output.splitlines()
    # remove 5th and 6th line and replace with header_line
    latex_output = latex_output[:4] + [header_line] + latex_output[6:]
    # remove last cline
    latex_output = latex_output[:-4] + latex_output[-3:]

    latex_output = "\n".join(latex_output)

    print(latex_output)
    print("\n\n")

# %% [markdown]
# # Box plots

# %% [markdown]
# ## Pct

# %%
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

metrics_rename = [
    "Improv. ARI (%)",
    "Improv. Silhouette (%)",
    "Improv. Calinski (%)",
]
agg_mode = None
df_metrics = get_metrics(df, model_groups, hpo_metrics, metrics_rename, agg_mode=agg_mode)
agg = "median"
group_by = ["dataset_name", "model_collab", "hpo_seed"]
df_metrics = df_metrics.drop(columns=["agent_i"])
df_metrics = df_metrics.groupby(group_by).agg([agg])
df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
df_metrics = df_metrics.reset_index()

metrics_re_rename = {f"{metric} {agg}": f"{agg.capitalize()} {metric}" for metric in metrics_rename}
df_metrics = df_metrics.rename(columns=metrics_re_rename)

# %%
order = [
    "CoresetKMeans",
    "DistributedKMeans",
    "DPFMPS-P2Est",
    "DPFMPS",
    "VertCoHiRF",
    "R-VertCoHiRF",
    "VertCoHiRF-DBSCAN",
    "R-VertCoHiRF-DBSCAN",
    # "VertCoHiRF-KernelRBF",
    # "R-VertCoHiRF-KernelRBF",
    "VertCoHiRF-SC-SRGF",
]
datasets = [
    "alizadeh-2000-v2",
    "coil-20",
    "garber-2001",
    "nursery",
    "shuttle",
]
metrics = [f"{agg.capitalize()} Improv. ARI (%)" , f"{agg.capitalize()} Improv. Silhouette (%)"]
for dataset in datasets:
    df_box = df_metrics.loc[df_metrics["dataset_name"] == dataset]
    # hide dbscan for alizadeh-2000-v2 and garber-2001 (messing up the scale)
    # hide sc-srgf for nursery and shuttle (dont run because of memory issues)
    if dataset in ["alizadeh-2000-v2", "garber-2001"]:
        order_ = order.copy()
        order_.remove("VertCoHiRF-DBSCAN")
        order_.remove("R-VertCoHiRF-DBSCAN")
    elif dataset in ["nursery", "shuttle"]:
        order_ = order.copy()
        order_.remove("VertCoHiRF-SC-SRGF")
    else:
        order_ = order
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
        fig, axs = plt.subplots(1, 2, figsize=(12 * fig_scale * cm, 7 * fig_scale * cm))
        plt.subplots_adjust(wspace=0.05)
        axs = axs.flatten()
        for ax, metric in zip(axs, metrics):
            ax = sns.boxplot(
                data=df_box, y="model_collab", hue="model_collab", x=metric, order=order_, hue_order=order_, ax=ax
            )
            ax.set_ylabel("")
            # only keep y axis label for first plot
            if ax != axs[0]:
                ax.set_yticks([])
        plt.show()
        fig.savefig(results_dir / f"boxplot_improvement_pct_{dataset}_ari_silhouette.pdf", bbox_inches='tight', dpi=600)
        # plt.title(f"Improvement in Adjusted Rand Index on dataset {dataset}")

# %% [markdown]
# ## Diff

# %%
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

metrics_rename = [
    "Diff. ARI",
    "Diff. Silhouette",
    "Diff. Calinski",
]
agg_mode = None
df_metrics = get_metrics(df, model_groups, hpo_metrics, metrics_rename, agg_mode=agg_mode, improv="diff")
agg = "median"
group_by = ["dataset_name", "model_collab", "hpo_seed"]
df_metrics = df_metrics.drop(columns=["agent_i"])
df_metrics = df_metrics.groupby(group_by).agg([agg])
df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
df_metrics = df_metrics.reset_index()

metrics_re_rename = {f"{metric} {agg}": f"{agg.capitalize()} {metric}" for metric in metrics_rename}
df_metrics = df_metrics.rename(columns=metrics_re_rename)

# %%
order = [
    "CoresetKMeans",
    "DistributedKMeans",
    "DPFMPS-P2Est",
    "DPFMPS",
    "VertCoHiRF",
    "R-VertCoHiRF",
    "VertCoHiRF-DBSCAN",
    "R-VertCoHiRF-DBSCAN",
    # "VertCoHiRF-KernelRBF",
    # "R-VertCoHiRF-KernelRBF",
    "VertCoHiRF-SC-SRGF",
]
datasets = [
    "alizadeh-2000-v2",
    "coil-20",
    "garber-2001",
    "nursery",
    "shuttle",
]
metrics = [f"{agg.capitalize()} Diff. ARI", f"{agg.capitalize()} Diff. Silhouette"]
for dataset in datasets:
    df_box = df_metrics.loc[df_metrics["dataset_name"] == dataset]
    # hide dbscan for alizadeh-2000-v2 and garber-2001 (messing up the scale)
    # hide sc-srgf for nursery and shuttle (dont run because of memory issues)
    if dataset in ["alizadeh-2000-v2", "garber-2001"]:
        order_ = order.copy()
        order_.remove("VertCoHiRF-DBSCAN")
        order_.remove("R-VertCoHiRF-DBSCAN")
    elif dataset in ["nursery", "shuttle"]:
        order_ = order.copy()
        order_.remove("VertCoHiRF-SC-SRGF")
    else:
        order_ = order
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
        fig, axs = plt.subplots(1, 2, figsize=(12 * fig_scale * cm, 7 * fig_scale * cm))
        plt.subplots_adjust(wspace=0.05)
        axs = axs.flatten()
        for ax, metric in zip(axs, metrics):
            ax = sns.boxplot(
                data=df_box, y="model_collab", hue="model_collab", x=metric, order=order_, hue_order=order_, ax=ax
            )
            ax.set_ylabel("")
            # only keep y axis label for first plot
            if ax != axs[0]:
                ax.set_yticks([])
        plt.show()
        fig.savefig(results_dir / f"boxplot_improvement_diff_{dataset}_ari_silhouette.pdf", bbox_inches="tight", dpi=600)
        # plt.title(f"Improvement in Adjusted Rand Index on dataset {dataset}")

# %% [markdown]
# ## Raw

# %%
model_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DBSCAN-50": "DBSCAN",
    "DistributedKMeans-50": "DistributedKMeans",
    "KMeans-50": "KMeans",
    "KernelRBFKMeans-50": "KernelRBFKMeans",
    "SpectralSubspaceRandomization-50": "SC-SRGF",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
    "VeCoHiRF-top-down-1iter-50": "R-VertCoHiRF",
}

dataset_names = {
    "binary_alpha_digits": "binary-alpha-digits",
    "mnist_784": "mnist",
}  # otherwise we get an error in latex

# Filter runs
df = df_runs_parents.copy()
df = df.loc[df["standardize"] == True]
df = df.loc[df["n_agents"] == 3]
df = df.loc[df["model"].isin(model_names.keys())]
df = df.loc[df["hpo_seed"].isin(range(5))]
df = df.replace({"model": model_names})
df = df.replace({"dataset_name": dataset_names})
df = df[
    [
        "model",
        "dataset_name",
        "agent_i",
        "hpo_seed",
        "hpo_metric",
        "best/adjusted_rand_mean",
        "best/silhouette_mean",
        "best/calinski_harabasz_score_mean",
    ]
]

# define model groups
model_groups = {
    "KMeans": [
        "KMeans",
        # "KernelKMeans",
        # "DBSCAN",
        # "SC-SRGF",
        "CoresetKMeans",
        "DistributedKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "R-VertCoHiRF",
        "VertCoHiRF-KernelRBF",
        "R-VertCoHiRF-KernelRBF",
        "VertCoHiRF-DBSCAN",
        "R-VertCoHiRF-DBSCAN",
        "VertCoHiRF-SC-SRGF",
    ],
}

# %%
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

metrics_rename = [
    "Diff. ARI",
    "Diff. Silhouette",
    "Diff. Calinski",
]
metrics_rename_2 = [
	"ARI",
	"Silhouette",
	"Calinski"
]
agg_mode = None
df_metrics = get_metrics(df, model_groups, hpo_metrics, metrics_rename, agg_mode=agg_mode, improv="diff", raw_metrics=True)
agg = "first"  # same for all agent_i
group_by = ["dataset_name", "model_collab", "hpo_seed"]
df_metrics = df_metrics.drop(columns=["agent_i"])
df_metrics = df_metrics.groupby(group_by).agg([agg])
df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
df_metrics = df_metrics.reset_index()

metrics_re_rename = {f"best/{hpo_metric}_collab {agg}": metric for hpo_metric, metric in zip(hpo_metrics, metrics_rename_2)}
df_metrics = df_metrics.rename(columns=metrics_re_rename)

# %%
df_metrics

# %%
order = [
    "CoresetKMeans",
    "DistributedKMeans",
    "DPFMPS-P2Est",
    "DPFMPS",
    "VertCoHiRF",
    "R-VertCoHiRF",
    "VertCoHiRF-DBSCAN",
    "R-VertCoHiRF-DBSCAN",
    # "VertCoHiRF-KernelRBF",
    # "R-VertCoHiRF-KernelRBF",
    "VertCoHiRF-SC-SRGF",
]
datasets = [
    "alizadeh-2000-v2",
    "coil-20",
    "garber-2001",
    "nursery",
    "shuttle",
]
metrics = [f"ARI", f"Silhouette"]
for dataset in datasets:
    df_box = df_metrics.loc[df_metrics["dataset_name"] == dataset]
    # hide dbscan for alizadeh-2000-v2 and garber-2001 (messing up the scale)
    # hide sc-srgf for nursery and shuttle (dont run because of memory issues)
    if dataset in ["alizadeh-2000-v2", "garber-2001"]:
        order_ = order.copy()
        order_.remove("VertCoHiRF-DBSCAN")
        order_.remove("R-VertCoHiRF-DBSCAN")
    elif dataset in ["nursery", "shuttle"]:
        order_ = order.copy()
        order_.remove("VertCoHiRF-SC-SRGF")
    else:
        order_ = order
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
        fig, axs = plt.subplots(1, 2, figsize=(12 * fig_scale * cm, 7 * fig_scale * cm))
        plt.subplots_adjust(wspace=0.05)
        axs = axs.flatten()
        for ax, metric in zip(axs, metrics):
            ax = sns.boxplot(
                data=df_box, y="model_collab", hue="model_collab", x=metric, order=order_, hue_order=order_, ax=ax
            )
            ax.set_ylabel("")
            # only keep y axis label for first plot
            if ax != axs[0]:
                ax.set_yticks([])
        plt.show()
        fig.savefig(
            results_dir / f"boxplot_improvement_raw_{dataset}_ari_silhouette.pdf", bbox_inches="tight", dpi=600
        )
        # plt.title(f"Improvement in Adjusted Rand Index on dataset {dataset}")

# %% [markdown]
# ## Katia

# %% [markdown]
# ## ARI

# %%
model_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DBSCAN-50": "DBSCAN",
    "DistributedKMeans-50": "DistributedKMeans",
    "KMeans-50": "KMeans",
    "KernelRBFKMeans-50": "KernelRBFKMeans",
    "SpectralSubspaceRandomization-50": "SC-SRGF",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
    "VeCoHiRF-top-down-1iter-50": "R-VertCoHiRF",
}

dataset_names = {
    "binary_alpha_digits": "binary-alpha-digits",
    "mnist_784": "mnist",
}  # otherwise we get an error in latex

# Filter runs
df = df_runs_parents.copy()
df = df.loc[df["standardize"] == True]
df = df.loc[df["n_agents"] == 3]
df = df.loc[df["model"].isin(model_names.keys())]
df = df.loc[df["hpo_seed"].isin(range(5))]
df = df.replace({"model": model_names})
df = df.replace({"dataset_name": dataset_names})
df = df[
    [
        "model",
        "dataset_name",
        "agent_i",
        "hpo_seed",
        "hpo_metric",
        "best/adjusted_rand_mean",
        "best/silhouette_mean",
        "best/calinski_harabasz_score_mean",
    ]
]

# define model groups
model_groups = {
    "KMeans": [
        "KMeans",
        # "KernelKMeans",
        # "DBSCAN",
        # "SC-SRGF",
        "CoresetKMeans",
        "DistributedKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "R-VertCoHiRF",
        "VertCoHiRF-KernelRBF",
        "R-VertCoHiRF-KernelRBF",
        "VertCoHiRF-DBSCAN",
        "R-VertCoHiRF-DBSCAN",
        "VertCoHiRF-SC-SRGF",
    ],
}

# %%
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

metrics_rename = [
    "Diff. ARI",
    "Diff. Silhouette",
    "Diff. Calinski",
]
metrics_rename_2 = ["ARI", "Silhouette", "Calinski"]
agg_mode = None
df_metrics = get_metrics(
    df, model_groups, hpo_metrics, metrics_rename, agg_mode=agg_mode, improv="diff", raw_metrics=True
)
agg = "first"  # same for all agent_i
group_by = ["dataset_name", "model_collab", "hpo_seed"]
df_metrics = df_metrics.drop(columns=["agent_i"])
df_metrics = df_metrics.groupby(group_by).agg([agg])
df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
df_metrics = df_metrics.reset_index()

metrics_re_rename = {
    f"best/{hpo_metric}_collab {agg}": metric for hpo_metric, metric in zip(hpo_metrics, metrics_rename_2)
}
df_metrics = df_metrics.rename(columns=metrics_re_rename)


# %%
def plot_katia_boxplot(df_metrics, dataset_name, model_names, order, metric, reference_metric, 
                        results_dir, suffix=""):
    """
    Helper function to create improved boxplot figures for Katia's article.
    
    Parameters:
    -----------
    df_metrics : DataFrame
        DataFrame with metrics data
    dataset_name : str
        Name of the dataset
    model_names : dict
        Dictionary mapping internal model names to display names
    order : list
        Order of models to display
    metric : str
        Metric to plot (e.g., "ARI", "Silhouette")
    reference_metric : str
        Reference metric column name for the vertical line
    results_dir : Path
        Directory to save the figure
    suffix : str, optional
        Suffix to add to filename
    show_legend : bool, optional
        Whether to show the legend
    """
    df_plot = df_metrics.copy()
    df_plot = df_plot.loc[df_plot["dataset_name"] == dataset_name]
    df_plot = df_plot.loc[df_plot["model_collab"].isin(model_names.keys())]
    df_plot = df_plot.replace({"model_collab": model_names})
    df_box = df_plot.loc[df_plot["dataset_name"] == dataset_name]

    with mpl.rc_context(
        rc={
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times", "serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": False,
            "font.size": 64,
            "font.weight": "bold",  # Bold fonts for better visibility when scaled down
            "axes.linewidth": 2.5,  # Thicker axes for better visibility
            "axes.labelsize": 64,
            "axes.labelweight": "bold",
            "axes.titlesize": 64,
            "xtick.labelsize": 48,  # Slightly larger tick labels
            "ytick.labelsize": 48,
            "xtick.major.width": 2.0,  # Thicker tick marks
            "ytick.major.width": 2.0,
            "legend.fontsize": 36,  # Larger legend
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.shadow": False,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "black",
            "grid.alpha": 0.4,  # Slightly more subtle grid
            "axes.grid": True,
            "grid.linewidth": 1.0,  # Thicker grid lines
            "lines.linewidth": 3.0,  # Thicker lines
        }
    ):
        cm = 1 / 2.54  # centimeters to inches
        fig_scale = 3.0
        fig, ax = plt.subplots(figsize=(12 * fig_scale * cm, 7 * fig_scale * cm))
        plt.subplots_adjust(left=0.25, right=0.98, top=0.98, bottom=0.15)  # Tighter margins
        reference_value = df_box.loc[df_box["model_collab"] == order[0], reference_metric].mean()
        ax = sns.boxplot(
            data=df_box,
            y="model_collab",
            hue="model_collab",
            x=metric,
            order=order,
            hue_order=order,
            ax=ax,
            linewidth=2.5,
            fliersize=8,  # Thicker box lines and larger outlier markers
        )
        ax.set_ylabel("", fontweight="bold")
        # Make y-tick labels bold
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.axvline(reference_value, color="blue", linestyle="--", linewidth=3.5, label="Local KMeans (median)")
        plt.show()

        # Create filename
        metric_name = metric.lower()
        filename = f"boxplot_{metric_name}_{dataset_name}{suffix}.pdf"
        fig.savefig(results_dir / filename, bbox_inches="tight", dpi=600)


# %%
# alizadeh-2000-v2 - ARI
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="alizadeh-2000-v2",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "VertCoHiRF-SC-SRGF": "VertCoHiRF-SC-SRGF",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-SC-SRGF",
    ],
    metric="ARI",
    reference_metric="best/adjusted_rand_mean first",
    results_dir=results_dir,
)

# %%
# chowdary-2006 - ARI
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="chowdary-2006",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "VertCoHiRF-SC-SRGF": "VertCoHiRF-SC-SRGF",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-SC-SRGF",
    ],
    metric="ARI",
    reference_metric="best/adjusted_rand_mean first",
    results_dir=results_dir,
)

# %%
# coil-20 - ARI
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="coil-20",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "R-VertCoHiRF-DBSCAN": "VertCoHiRF-DBSCAN",
        "VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-DBSCAN",
    ],
    metric="ARI",
    reference_metric="best/adjusted_rand_mean first",
    results_dir=results_dir,
)

# %%
# garber-2001 - ARI
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="garber-2001",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "VertCoHiRF-SC-SRGF": "VertCoHiRF-SC-SRGF",
        "VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-SC-SRGF",
    ],
    metric="ARI",
    reference_metric="best/adjusted_rand_mean first",
    results_dir=results_dir,
)

# %%
# nursery - ARI
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="nursery",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "R-VertCoHiRF-KernelRBF": "VertCoHiRF-KernelRBF",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-KernelRBF",
    ],
    metric="ARI",
    reference_metric="best/adjusted_rand_mean first",
    results_dir=results_dir,
)

# %%
# shuttle - ARI
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="shuttle",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "R-VertCoHiRF-DBSCAN": "VertCoHiRF-DBSCAN",
        "VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-DBSCAN",
    ],
    metric="ARI",
    reference_metric="best/adjusted_rand_mean first",
    results_dir=results_dir,
)

# %% [markdown]
# ## Silhouette

# %%
model_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DBSCAN-50": "DBSCAN",
    "DistributedKMeans-50": "DistributedKMeans",
    "KMeans-50": "KMeans",
    "KernelRBFKMeans-50": "KernelRBFKMeans",
    "SpectralSubspaceRandomization-50": "SC-SRGF",
    "V2way-50": "DPFMPS-P2Est",
    "VPC-50": "DPFMPS",
    "VeCoHiRF-1iter-50": "VertCoHiRF",
    "VeCoHiRF-DBSCAN-1iter-50": "VertCoHiRF-DBSCAN",
    "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VertCoHiRF-DBSCAN",
    "VeCoHiRF-KernelRBF-1iter-50": "VertCoHiRF-KernelRBF",
    "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VertCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VertCoHiRF-SC-SRGF",
    "VeCoHiRF-top-down-1iter-50": "R-VertCoHiRF",
}

dataset_names = {
    "binary_alpha_digits": "binary-alpha-digits",
    "mnist_784": "mnist",
}  # otherwise we get an error in latex

# Filter runs
df = df_runs_parents.copy()
df = df.loc[df["standardize"] == True]
df = df.loc[df["n_agents"] == 3]
df = df.loc[df["model"].isin(model_names.keys())]
df = df.loc[df["hpo_seed"].isin(range(5))]
df = df.replace({"model": model_names})
df = df.replace({"dataset_name": dataset_names})
df = df[
    [
        "model",
        "dataset_name",
        "agent_i",
        "hpo_seed",
        "hpo_metric",
        "best/adjusted_rand_mean",
        "best/silhouette_mean",
        "best/calinski_harabasz_score_mean",
    ]
]

# define model groups
model_groups = {
    "KMeans": [
        "KMeans",
        # "KernelKMeans",
        # "DBSCAN",
        # "SC-SRGF",
        "CoresetKMeans",
        "DistributedKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "R-VertCoHiRF",
        "VertCoHiRF-KernelRBF",
        "R-VertCoHiRF-KernelRBF",
        "VertCoHiRF-DBSCAN",
        "R-VertCoHiRF-DBSCAN",
        "VertCoHiRF-SC-SRGF",
    ],
}

# %%
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

metrics_rename = [
    "Diff. ARI",
    "Diff. Silhouette",
    "Diff. Calinski",
]
metrics_rename_2 = ["ARI", "Silhouette", "Calinski"]
agg_mode = None
df_metrics = get_metrics(
    df, model_groups, hpo_metrics, metrics_rename, agg_mode=agg_mode, improv="diff", raw_metrics=True
)
agg = "first"  # same for all agent_i
group_by = ["dataset_name", "model_collab", "hpo_seed"]
df_metrics = df_metrics.drop(columns=["agent_i"])
df_metrics = df_metrics.groupby(group_by).agg([agg])
df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
df_metrics = df_metrics.reset_index()

metrics_re_rename = {
    f"best/{hpo_metric}_collab {agg}": metric for hpo_metric, metric in zip(hpo_metrics, metrics_rename_2)
}
df_metrics = df_metrics.rename(columns=metrics_re_rename)

# %%
# alizadeh-2000-v2 - Silhouette
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="alizadeh-2000-v2",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "VertCoHiRF-SC-SRGF": "VertCoHiRF-SC-SRGF",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-SC-SRGF",
    ],
    metric="Silhouette",
    reference_metric="best/silhouette_mean first",
    results_dir=results_dir,
)

# %%
# chowdary-2006 - Silhouette
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="chowdary-2006",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "VertCoHiRF-SC-SRGF": "VertCoHiRF-SC-SRGF",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-SC-SRGF",
    ],
    metric="Silhouette",
    reference_metric="best/silhouette_mean first",
    results_dir=results_dir,
)

# %%
# coil-20 - Silhouette
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="coil-20",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "R-VertCoHiRF-DBSCAN": "VertCoHiRF-DBSCAN",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-DBSCAN",
    ],
    metric="Silhouette",
    reference_metric="best/silhouette_mean first",
    results_dir=results_dir,
)

# %%
# garber-2001 - Silhouette
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="garber-2001",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "VertCoHiRF-SC-SRGF": "VertCoHiRF-SC-SRGF",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-SC-SRGF",
    ],
    metric="Silhouette",
    reference_metric="best/silhouette_mean first",
    results_dir=results_dir,
)

# %%
# nursery - Silhouette
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="nursery",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "R-VertCoHiRF-KernelRBF": "VertCoHiRF-KernelRBF",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-KernelRBF",
    ],
    metric="Silhouette",
    reference_metric="best/silhouette_mean first",
    results_dir=results_dir,
)

# %%
# shuttle - Silhouette
plot_katia_boxplot(
    df_metrics=df_metrics,
    dataset_name="shuttle",
    model_names={
        "CoresetKMeans": "CoresetKMeans",
        "DistributedKMeans": "DistributedKMeans",
        "DPFMPS-P2Est": "DPFMPS-P2Est",
        "DPFMPS": "DPFMPS",
        "VertCoHiRF-DBSCAN": "VertCoHiRF-DBSCAN",
        "R-VertCoHiRF": "VertCoHiRF",
    },
    order=[
        "DistributedKMeans",
        "CoresetKMeans",
        "DPFMPS-P2Est",
        "DPFMPS",
        "VertCoHiRF",
        "VertCoHiRF-DBSCAN",
    ],
    metric="Silhouette",
    reference_metric="best/silhouette_mean first",
    results_dir=results_dir,
)


# %% [markdown]
# # Katia tables

# %%
def get_metrics(df, model_groups, hpo_metrics, metrics_rename, agg_mode="mean", improv="pct", raw_metrics=False):
    df["model_group"] = df["model"].apply(
        lambda x: next((group for group, models in model_groups.items() if x in models), "Other")
    )
    join_columns = ["dataset_name", "hpo_seed", "hpo_metric", "model_group"]
    df_base_models = df.loc[df["model"].isin(model_groups.keys())].copy()
    df_not_base_models = df.loc[~df["model"].isin(model_groups.keys())].copy()
    df_not_base_models = df_not_base_models.set_index(join_columns)
    df_not_base_models.drop(columns=["agent_i"], inplace=True)
    # each row in df_base_models is "exploded" into multiple rows for each collaborative model
    df_collab = df_base_models.join(df_not_base_models, on=join_columns, rsuffix="_collab")
    # for each metric, compute the improvement of each collaborative model over the base model
    if improv == "pct":
        for hpo_metric in hpo_metrics:
            df_collab[f"{improv}_improvement/{hpo_metric}"] = (
                (df_collab[f"best/{hpo_metric}_collab"] - df_collab[f"best/{hpo_metric}"])
                / df_collab[f"best/{hpo_metric}"]
                * 100
            )
    elif improv == "diff":
        for hpo_metric in hpo_metrics:
            df_collab[f"{improv}_improvement/{hpo_metric}"] = (
                df_collab[f"best/{hpo_metric}_collab"] - df_collab[f"best/{hpo_metric}"]
            )
    else:
        raise ValueError(f"Unknown improv method: {improv}")

    dfs_metrics = {}

    for hpo_metric, metric_rename in zip(hpo_metrics, metrics_rename):
        columns_to_keep = ["dataset_name", "model_collab", "hpo_seed", "agent_i", f"{improv}_improvement/{hpo_metric}"]
        if raw_metrics:
            columns_to_keep += [f"best/{hpo_metric}_collab", f"best/{hpo_metric}"]
        df_metric = df_collab.loc[df_collab["hpo_metric"] == hpo_metric][columns_to_keep].rename(
            columns={f"{improv}_improvement/{hpo_metric}": metric_rename}
        )
        df_metric = df_metric.dropna(subset=[metric_rename])
        df_metric = df_metric.set_index(["dataset_name", "model_collab", "hpo_seed", "agent_i"])
        df_metric = df_metric.astype({metric_rename: float})
        dfs_metrics[metric_rename] = df_metric

    df_metrics = pd.concat(dfs_metrics.values(), axis=1, join="outer")
    df_metrics = df_metrics.reset_index()

    if agg_mode is not None:
        # aggregate for each dataset_name, model_collab and hpo_seed then do the mean and std across hpo_seeds

        # drop columns where we are going to aggregate
        df_metrics = df_metrics.drop(columns=["agent_i"])
        # aggregate
        df_metrics = df_metrics.groupby(["dataset_name", "model_collab", "hpo_seed"]).agg([agg_mode])
        # flatten multiindex columns
        df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
        # reset index
        df_metrics = df_metrics.reset_index()
        # mean across hpo_seed
        df_metrics = df_metrics.drop(columns=["hpo_seed"])
        df_metrics = df_metrics.groupby(["dataset_name", "model_collab"]).agg(["mean", "std"])
        df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
        # Rename index levels
        df_metrics.index.names = ["Dataset", "Model"]

        for metric in metrics_rename:
            df_metrics[f"{metric}"] = (
                df_metrics[f"{metric} {agg_mode} mean"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
                + " $\\pm$ "
                + df_metrics[f"{metric} {agg_mode} std"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
            )
        
        if raw_metrics:
            for hpo_metric in hpo_metrics:
                df_metrics[f"{hpo_metric}_collab"] = (
                    df_metrics[f"best/{hpo_metric}_collab {agg_mode} mean"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
                    + " $\\pm$ "
                    + df_metrics[f"best/{hpo_metric}_collab {agg_mode} std"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
                )
                df_metrics[f"{hpo_metric}_base"] = (
                    df_metrics[f"best/{hpo_metric} {agg_mode} mean"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
                    + " $\\pm$ "
                    + df_metrics[f"best/{hpo_metric} {agg_mode} std"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
                )

    return df_metrics


# %%
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

metrics_rename = [
    "Diff. ARI",
    "Diff. Silhouette",
    "Diff. Calinski",
]
agg_mode = "median"
df_metrics = get_metrics(df, model_groups, hpo_metrics, metrics_rename, agg_mode=agg_mode, improv="diff", raw_metrics=True)
# agg = "median"
# group_by = ["dataset_name", "model_collab", "hpo_seed"]
# df_metrics = df_metrics.drop(columns=["agent_i"])
# df_metrics = df_metrics.groupby(group_by).agg([agg])
# df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
# df_metrics = df_metrics.reset_index()

# metrics_re_rename = {f"{metric} {agg}": f"{agg.capitalize()} {metric}" for metric in metrics_rename}
# df_metrics = df_metrics.rename(columns=metrics_re_rename)

# %%
df_metrics

# %%
df_latex

# %%
df_print

# %%
df_latex = df_metrics.copy()
df_latex = df_latex.reset_index()
metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
]
rename_metrics = [
    "ARI",
    "Silhouette",
]
for metric, rename_metric in zip(metrics, rename_metrics):
    print(f"% LaTeX table for metric: {rename_metric}")
    # print per dataset
    for dataset in df_latex["Dataset"].unique():
        df_print = df_latex.copy()
        df_print = df_print.loc[df_print["Dataset"] == dataset]
        df_print = df_print[["Model", f"{metric}_base", f"{metric}_collab", f"Diff. {rename_metric}",]]
        df_print = df_print.rename(columns={
            f"{metric}_base": f"Base {rename_metric}",
            f"{metric}_collab": f"Collab. {rename_metric}",
        })
        latex_output = df_print.to_latex(
            index=False,
            # hrules=True,
            # clines="skip-last;data",
            # convert_css=True,
            # column_format="l" * (len(df_print.columns)),
            # # environment="longtable",
            caption=f"Clustering results on dataset {dataset}",
        )
        print(latex_output)
        print("\n\n")
#     df_print = df_latex.copy()
#     df_print = df_print.loc[dataset]
#     columns_to_hide = [col for col in df_latex.columns if col not in (metrics_rename)]
#     df_print = df_print.style.hide(columns_to_hide, axis=1)
#     for col in metrics_rename:
#         highlight_metric = partial(highlight_max, column_name=f"{col} {agg_mode} mean")
#         underline_2nd_metric = partial(underline_2nd_max, column_name=f"{col} {agg_mode} mean")
#         # if col in ["Davies-Bouldin", "Best Time", "HPO Time"]:
#         #     highlight_metric = partial(highlight_min, column_name=f"{col} mean")
#         #     underline_2nd_metric = partial(underline_2nd_min, column_name=f"{col} mean")
#         (
#             df_print.apply(highlight_metric, subset=[col, f"{col} {agg_mode} mean"], axis=None).apply(
#                 underline_2nd_metric, subset=[col, f"{col} {agg_mode} mean"], axis=None
#             )
#         )

#     latex_output = df_print.to_latex(
#         hrules=True,
#         clines="skip-last;data",
#         convert_css=True,
#         column_format="ll" + "l" * (len(df_print.columns) - len(columns_to_hide)),
#         # environment="longtable",
#         caption=f"Clustering results on dataset {dataset}",
#     )

#     # fix header
#     columns = df_print.index.names + [col for col in df_print.columns if col not in columns_to_hide]
#     header_line = " & ".join(columns) + r" \\"

#     # split into lines
#     latex_output = latex_output.splitlines()
#     # remove 5th and 6th line and replace with header_line
#     latex_output = latex_output[:4] + [header_line] + latex_output[6:]
#     # remove last cline
#     # latex_output = latex_output[:-4] + latex_output[-3:]

#     latex_output = "\n".join(latex_output)

#     print(latex_output)
#     print("\n\n")
#     print("\pagebreak")

# %% [markdown]
# # Save Results

# %% [markdown]
# ## Load mlflow runs

# %%
results_dir = Path.cwd().parent / "results" / "real" / "dauphine"
os.makedirs(results_dir, exist_ok=True)

# %%
db_port = 6001
db_name = "cocohirf"
url = f"postgresql://belucci@localhost:{db_port}/{db_name}"
engine = create_engine(url)
query = "SELECT experiments.name from experiments"
experiment_names = pd.read_sql(query, engine)["name"].tolist()

# %%
experiment_names

# %%
experiments_names = [exp for exp in experiment_names if exp.startswith("real-")]

# %%
experiments_names

# %%
query = "SELECT DISTINCT(key) FROM params WHERE key LIKE 'best/%%'"
best_params = pd.read_sql(query, engine)["key"].tolist()

# %%
params_columns = [
    "model",
    "model_alias",
    "dataset_id",
    "n_trials",
    "n_trials_1",
    "dataset_name",
    "standardize",
    "hpo_metric",
    "hpo_metric_2",
	"direction",
    "direction_2",
    "hpo_seed",
    "seed_dataset_order",
    "n_agents",
    "agent_i",
] + best_params

# %%
metrics = [
	"rand_score",
	"adjusted_rand",
	"mutual_info",
	"adjusted_mutual_info",
	"normalized_mutual_info",
	"homogeneity",
	"completeness",
	"v_measure",
	"silhouette",
	"calinski_harabasz_score",
	"davies_bouldin_score",
]
best_mean_metrics = [f"best/{metric}_mean" for metric in metrics]
best_std_metrics = [f"best/{metric}_std" for metric in metrics]

# %%
latest_metrics_columns = [
    "fit_model_return_elapsed_time",
    "max_memory_used_after_fit",
    "max_memory_used",
	"best/min_n_clusters",
    "best/elapsed_time",
]
latest_metrics_columns += best_mean_metrics + best_std_metrics

# %%
tags_columns = ["raised_exception", "EXCEPTION", "mlflow.parentRunId", "Last step finished", "hpo_stage"]

# %%
runs_columns = ['run_uuid', 'status', 'start_time', 'end_time']
experiments_columns = []
other_table = 'params'
other_table_keys = params_columns
df_tags = get_df_runs_from_mlflow_sql(engine, runs_columns=runs_columns, experiments_columns=experiments_columns, experiments_names=experiments_names, other_table='tags', other_table_keys=tags_columns)
df_params = get_df_runs_from_mlflow_sql(engine, runs_columns=['run_uuid'], experiments_columns=experiments_columns, experiments_names=experiments_names, other_table=other_table, other_table_keys=other_table_keys)
df_latest_metrics = get_df_runs_from_mlflow_sql(engine, runs_columns=['run_uuid'], experiments_columns=experiments_columns, experiments_names=experiments_names, other_table='latest_metrics', other_table_keys=latest_metrics_columns)


# %%
dataset_characteristics = pd.read_csv(results_dir / "datasets_characteristics.csv", index_col=0)
dataset_characteristics.index = dataset_characteristics["openml_id"].astype(str)

# %%
df_runs_raw = df_tags.join(df_latest_metrics)
df_runs_raw = df_runs_raw.join(df_params)
df_runs_raw = df_runs_raw.join(dataset_characteristics, on='dataset_id', rsuffix='_dataset')
df_runs_raw.to_csv(results_dir / 'df_runs_raw.csv', index=True)

# %% [markdown]
# # Raw performance per dataset

# %%
model_names = {
    "CoresetKMeans-50": "CoresetKMeans",
    "DBSCAN-50": "DBSCAN",
    "DistributedKMeans-50": "DistributedKMeans",
    "KMeans-50": "KMeans",
    "KernelRBFKMeans-50": "KernelRBFKMeans",
    "SpectralSubspaceRandomization-50": "SC-SRGF",
    "V2way-50": "V2way",
    "VPC-50": "VPC",
    "VeCoHiRF-1iter-50": "VeCoHiRF",
    "VeCoHiRF-DBSCAN-1iter-50": "VeCoHiRF-DBSCAN",
    "VeCoHiRF-DBSCAN-top-down-1iter-50": "R-VeCoHiRF-DBSCAN",
    "VeCoHiRF-KernelRBF-1iter-50": "VeCoHiRF-KernelRBF",
    "VeCoHiRF-KernelRBF-top-down-1iter-50": "R-VeCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-1R-1iter-50": "VeCoHiRF-SC-SRGF",
    "VeCoHiRF-top-down-1iter-50": "R-VeCoHiRF",
}

dataset_names = {
    "binary_alpha_digits": "binary-alpha-digits",
    "mnist_784": "mnist",
}  # otherwise we get an error in latex

# Filter runs
df = df_runs_parents.copy()
df = df.loc[df["standardize"] == True]
df = df.loc[df["n_agents"] == 3]
df = df.loc[df["model"].isin(model_names.keys())]
df = df.loc[df["hpo_seed"].isin(range(5))]
df = df.replace({"model": model_names})
df = df.replace({"dataset_name": dataset_names})
df = df[
    [
        "model",
        "dataset_name",
        "agent_i",
        "hpo_seed",
        "hpo_metric",
        "best/adjusted_rand_mean",
        "best/silhouette_mean",
        "best/calinski_harabasz_score_mean",
    ]
]

# %%
hpo_metrics = [
    "adjusted_rand_mean",
    "silhouette_mean",
    "calinski_harabasz_score_mean",
]

hpo_metrics_rename = [
    "ARI",
    "Silhouette",
    "Calinski",
]

dfs_metrics = {}

for hpo_metric, hpo_metric_rename in zip(hpo_metrics, hpo_metrics_rename):
    original_metric = hpo_metric
    df_metric = df.loc[df["hpo_metric"] == original_metric][
        ["dataset_name", "model", "hpo_seed", "agent_i", f"best/{hpo_metric}"]
    ].rename(columns={f"best/{hpo_metric}": hpo_metric_rename})
    df_metric = df_metric.dropna(subset=[hpo_metric_rename])
    df_metric = df_metric.set_index(["dataset_name", "model", "hpo_seed", "agent_i"])
    df_metric = df_metric.astype({hpo_metric_rename: float})
    dfs_metrics[hpo_metric_rename] = df_metric

df_metrics = pd.concat(dfs_metrics.values(), axis=1, join="outer")
df_metrics = df_metrics.reset_index()
df_metrics = df_metrics.drop(columns=["agent_i"])

# calculate mean and std
df_metrics = df_metrics.groupby(["dataset_name", "model"]).agg(["mean", "std"])
# flatten multiindex columns
df_metrics.columns = [" ".join(col).strip() for col in df_metrics.columns.values]
# drop hpo_seed level
df_metrics = df_metrics.drop(columns=["hpo_seed mean", "hpo_seed std"])
# Rename index levels
df_metrics.index.names = ["Dataset", "Model"]


for metric in hpo_metrics_rename:
    df_metrics[f"{metric}"] = (
        df_metrics[f"{metric} mean"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
        + " $\\pm$ "
        + df_metrics[f"{metric} std"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
    )

# %%
df_latex

# %%
metrics_rename = [
    "ARI",
    "Silhouette",
    "Calinski",
]
df_latex = df_metrics.copy()
# df_latex = df_latex.reset_index()
# # reapply model groups
# df_latex["Base Model"] = df_latex["Model"].apply(
#     lambda x: next((group for group, models in model_groups.items() if x in models), "Other")
# )
# # redefine index with model_group
# df_latex = df_latex.set_index(["Dataset", "Base Model", "Model"])
# # sort by dataset, model_group, model
# df_latex = df_latex.sort_index(level=["Dataset", "Base Model", "Model"])
# raise
# print per dataset
for dataset in df_latex.index.get_level_values("Dataset").unique():
    df_print = df_latex.copy()
    df_print = df_print.loc[dataset]
    columns_to_hide = [col for col in df_latex.columns if col not in metrics_rename]
    df_print = df_print.style.hide(columns_to_hide, axis=1)
    for col in metrics_rename:
        highlight_metric = partial(highlight_max, column_name=f"{col} mean", level=None)
        underline_2nd_metric = partial(underline_2nd_max, column_name=f"{col} mean", level=None)
        # if col in ["Davies-Bouldin", "Best Time", "HPO Time"]:
        #     highlight_metric = partial(highlight_min, column_name=f"{col} mean")
        #     underline_2nd_metric = partial(underline_2nd_min, column_name=f"{col} mean")
        (
            df_print.apply(highlight_metric, subset=[col, f"{col} mean"], axis=None).apply(
                underline_2nd_metric, subset=[col, f"{col} mean"], axis=None
            )
        )

    latex_output = df_print.to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_print.columns) - len(columns_to_hide)),
        # environment="longtable",
        caption=f"Clustering results on dataset {dataset}",
    )

    # fix header
    columns = df_print.index.names + [col for col in df_print.columns if col not in columns_to_hide]
    header_line = " & ".join(columns) + r" \\"

    # split into lines
    latex_output = latex_output.splitlines()
    # remove 5th and 6th line and replace with header_line
    latex_output = latex_output[:4] + [header_line] + latex_output[6:]
    # remove last cline
    # latex_output = latex_output[:-4] + latex_output[-3:]

    latex_output = "\n".join(latex_output)

    print(latex_output)
    print("\n\n")
    print("\pagebreak")

# %% [markdown]
# # Debug and explore

# %%
# Create a mapping of run_id to parent_id
parent_map = df_runs_raw.set_index(df_runs_raw.index)["mlflow.parentRunId"].to_dict()

# Function to find the root parent
def find_root_parent(run_id, parent_map, max_depth=100):
    """Find the root parent by following the parent chain"""
    current_id = run_id
    depth = 0

    while current_id in parent_map and pd.notna(parent_map[current_id]) and depth < max_depth:
        current_id = parent_map[current_id]
        depth += 1

    return current_id


df = df_runs_raw.copy()
# get only children runs
df = df.loc[~df["mlflow.parentRunId"].isna()]
# get the top parent run for each child
df["root_parent_id"] = df.index.map(lambda x: find_root_parent(x, parent_map))
# filter only failed runs
runs_failed = runs_failed.loc[(runs_failed["raised_exception"] == True) | (runs_failed["status"] != "FINISHED")]
# count how many children runs failed for each parent
failed_counts = runs_failed.groupby("root_parent_id").size().reset_index(name="failed_children_count")
failed_counts = failed_counts.loc[failed_counts["failed_children_count"] > 5]
# # get parent runs without any child run (should have been deleted)
# df_parents = df_runs_raw.loc[df_runs_raw["mlflow.parentRunId"].isna()]
# parent_ids_with_children = df["root_parent_id"].unique()
# df_parents_no_children = df_parents.loc[~df_parents.index.isin(parent_ids_with_children)]
# df_parents_no_children

# %%
runs_to_delete_parents = list(failed_counts["root_parent_id"].unique())

# %%
# get children of runs iteratively
runs_to_delete = runs_to_delete_parents.copy()
while 1:
    df = df_runs_raw.copy()
    df = df.loc[df["mlflow.parentRunId"].isin(runs_to_delete_parents)]
    runs_to_delete_children = list(df.index)
    runs_to_delete = runs_to_delete + runs_to_delete_children
    runs_to_delete_parents = runs_to_delete_children
    if len(runs_to_delete_parents) == 0:
        break

# %%
len(runs_to_delete)

# %%
run_uuid_query = [f"'{run_id}'" for run_id in runs_to_delete]
run_uuid_query = ', '.join(run_uuid_query)

# %%
print(run_uuid_query)

# %%
query = f"""
UPDATE runs
SET lifecycle_stage = 'deleted'
WHERE run_uuid IN ({run_uuid_query}) 
"""
with engine.begin() as conn:
    conn.execute(text(query))

# %%
query = f"""
DELETE
FROM
	experiment_tags
WHERE
	experiment_id = ANY(
	SELECT
		experiment_id
	FROM
		experiments
	WHERE
		lifecycle_stage = 'deleted');

DELETE
FROM
	latest_metrics
WHERE
	run_uuid = ANY(
	SELECT
		run_uuid
	FROM
		runs
	WHERE
		lifecycle_stage = 'deleted');
	
DELETE
FROM
	metrics
WHERE
	run_uuid = ANY(
	SELECT
		run_uuid
	FROM
		runs
	WHERE
		lifecycle_stage = 'deleted');
	
DELETE
FROM
	params
WHERE
	run_uuid = ANY(
	SELECT
		run_uuid
	FROM
		runs
	WHERE
		lifecycle_stage = 'deleted');

DELETE
FROM
	tags
WHERE
	run_uuid = ANY(
	SELECT
		run_uuid
	FROM
		runs
	WHERE
		lifecycle_stage = 'deleted');
	
DELETE 
FROM 
	runs
WHERE 
	lifecycle_stage = 'deleted';

DELETE 
FROM 
	experiments
WHERE 
	lifecycle_stage = 'deleted';
"""
with engine.begin() as conn:
    conn.execute(text(query))

# %%
