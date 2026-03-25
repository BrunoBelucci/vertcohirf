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
from ml_experiments.analyze import get_df_runs_from_mlflow_sql, get_missing_entries
from pathlib import Path
import os
import pickle
from functools import partial

# %% [markdown]
# # Save Results

# %% [markdown]
# ## Load mlflow runs

# %%
results_dir = Path.cwd().parent / "results" / "real"
os.makedirs(results_dir, exist_ok=True)

# %%
db_port = 6101
db_name = "cocohirf"
url = f"postgresql://beluccib@localhost:{db_port}/{db_name}"
# url = f"postgresql://beluccib@clust5:{db_port}/{db_name}"
engine = create_engine(url)
query = "SELECT experiments.name from experiments"
experiment_names = pd.read_sql(query, engine)["name"].tolist()

# %%
experiment_names

# %%
experiments_names = [exp for exp in experiment_names if not exp.startswith("real-") or exp == 'Default']

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
tags_columns = [
    'raised_exception',
    'EXCEPTION',
    'mlflow.parentRunId',
	'Last step finished'
]

# %%
runs_columns = ['run_uuid', 'status', 'start_time', 'end_time']
experiments_columns = []
other_table = 'params'
other_table_keys = params_columns
df_params = get_df_runs_from_mlflow_sql(engine, runs_columns=runs_columns, experiments_columns=experiments_columns, experiments_names=experiments_names, other_table=other_table, other_table_keys=other_table_keys)
df_latest_metrics = get_df_runs_from_mlflow_sql(engine, runs_columns=['run_uuid'], experiments_columns=experiments_columns, experiments_names=experiments_names, other_table='latest_metrics', other_table_keys=latest_metrics_columns)
df_tags = get_df_runs_from_mlflow_sql(engine, runs_columns=['run_uuid'], experiments_columns=experiments_columns, experiments_names=experiments_names, other_table='tags', other_table_keys=tags_columns)

# %%
dataset_characteristics = pd.read_csv(results_dir / "datasets_characteristics.csv", index_col=0)
dataset_characteristics.index = dataset_characteristics["openml_id"].astype(str)

# %%
df_runs_raw = df_params.join(df_latest_metrics)
df_runs_raw = df_runs_raw.join(df_tags)
df_runs_raw = df_runs_raw.join(dataset_characteristics, on='dataset_id', rsuffix='_dataset')
df_runs_raw.to_csv(results_dir / 'df_runs_raw_tgcc.csv', index=True)

# %%
df_runs_raw = pd.read_csv(results_dir / "df_runs_raw_tgcc.csv", index_col=0)
df_runs_raw.loc[df_runs_raw["model"].isna(), "model"] = df_runs_raw.loc[df_runs_raw["model"].isna(), "model_alias"]
df_runs_raw.loc[df_runs_raw["n_trials"].isna(), "n_trials"] = df_runs_raw.loc[df_runs_raw["n_trials"].isna(), "n_trials_1"]
# df_runs_raw["n_trials"] = df_runs_raw["n_trials"].astype(int)
df_runs_raw.loc[df_runs_raw["hpo_metric"].isna(), "hpo_metric"] = df_runs_raw.loc[df_runs_raw["hpo_metric"].isna(), "hpo_metric_2"]
# mask = df_runs_raw["model"].str.contains("CoHiRF")
# df_runs_raw.loc[mask, "model"] = df_runs_raw.loc[mask].apply(lambda row: f"{row['model']}-{row['n_trials']}", axis=1)
df_runs_raw_parents = df_runs_raw.copy()
df_runs_raw_parents = df_runs_raw_parents.loc[df_runs_raw_parents["mlflow.parentRunId"].isna()]
df_runs_raw_parents["n_trials"] = df_runs_raw_parents["n_trials"].astype(int)
df_runs_raw_parents["model"] = df_runs_raw_parents["model"] + "-" + df_runs_raw_parents["n_trials"].astype(str)

# %%
df_runs_raw_parents.head(5)

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
# get number of children runs that raised exception for each parent run
children_exceptions = df_runs_raw.groupby("mlflow.parentRunId")["raised_exception"].sum()
df_runs_parents["n_children_raised_exception"] = df_runs_parents.index.map(children_exceptions).fillna(0)

# %%
df_runs_parents.loc[(df_runs_parents["n_children_raised_exception"] > 0) & (df_runs_parents["raised_exception"] == False) & (df_runs_parents["model"].str.find("SC-SRGF") == -1), ["dataset_id", "model", "hpo_metric", "n_children_raised_exception"]]

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
# Too memory intensive
dataset_ids_to_complete = [182, 1478, 1568]
model_names = ["VeCoHiRF-SC-SRGF-50"]
for dataset_id in dataset_ids_to_complete:
    for agent_i in agents_i:
        for n_agent in n_agents:
            for model_name in model_names:
                for hpo_metric in hpo_metrics:
                    for std in standardize:
                        for seed in hpo_seed:
                            new_row = {
                                "dataset_id": dataset_id,
                                "model": model_name,
                                "hpo_metric": hpo_metric,
                                "standardize": std,
                                "hpo_seed": seed,
                                "n_agents": n_agent,
                                "agent_i": agent_i,
                            }
                            for col in fill_columns:
                                new_row[col] = fill_value
                            df_to_cat.append(new_row)

# %%
df_runs_parents = pd.concat([df_runs_parents, pd.DataFrame(df_to_cat)], axis=0)

# %% [markdown]
# # Missing

# %%
model_nickname = df_runs_parents['model'].unique().tolist()
model_nickname.sort()
model_nickname

# %%
non_duplicate_columns = [
    "model",
    "dataset_id",
    "standardize",
    "hpo_metric",
    "hpo_seed",
    # "seed_dataset_order",
    "n_agents",
]
model_nickname = [
    "CoresetKMeans-50",
    "DistributedKMeans-50",
    # "DistributedKMeansLocal-50",
    "V2way-50",
    "VPC-50",
    # "VeCoHiRF-50",
    "VeCoHiRF-1iter-50",
	"VeCoHiRF-top-down-1iter-50",
    "VeCoHiRF-DBSCAN-50",
	"VeCoHiRF-DBSCAN-1iter-50",
	"VeCoHiRF-DBSCAN-top-down-1iter-50",
    "VeCoHiRF-KernelRBF-50",
    "VeCoHiRF-KernelRBF-1iter-50",
	"VeCoHiRF-KernelRBF-top-down-1iter-50",
	"VeCoHiRF-SC-SRGF-50",
	"VeCoHiRF-SC-SRGF-1R-50",
	"VeCoHiRF-SC-SRGF-2R-50",
]
dataset_id = [
    39,
    61,
    182,
    1478,
    1568,
    # 40685,
    40984,
    46773,
    46774,
    46775,
    46776,
    46777,
    46778,
    46779,
    46780,
    46781,
    46782,
    46783,
    # 554,
    # 1110,
    # 47039
]
standardize = [True]
hpo_metric = [
    "adjusted_rand_mean",
    # "adjusted_mutual_info",
    # "calinski_harabasz_score",
    # "normalized_mutual_info",
    # "davies_bouldin_score",
    # "silhouette",
]
hpo_seed = [i for i in range(5)]
n_agents = [2] 
columns_names = non_duplicate_columns
should_contain_values = [model_nickname, dataset_id, standardize, hpo_metric, hpo_seed, n_agents]
df_missing = get_missing_entries(df_runs_parents, columns_names, should_contain_values)
df_missing

# %%
non_duplicate_columns = [
    "model",
    "dataset_id",
    "standardize",
    "hpo_metric",
    "hpo_seed",
    # "seed_dataset_order",
    "n_agents",
    "agent_i",
]
model_nickname = [
    "KMeans-50",
]
dataset_id = [
    39,
    61,
    182,
    1478,
    1568,
    # 40685,
    40984,
    46773,
    46774,
    46775,
    46776,
    46777,
    46778,
    46779,
    46780,
    46781,
    46782,
    46783,
    # 554,
    # 1110,
    # 47039
]
standardize = [True]
hpo_metric = [
    "adjusted_rand_mean",
    # "adjusted_mutual_info",
    # "calinski_harabasz_score",
    # "normalized_mutual_info",
    # "davies_bouldin_score",
    # "silhouette",
]
hpo_seed = [i for i in range(5)]
n_agents = [2]
agent_i = [0, 1]
columns_names = non_duplicate_columns
should_contain_values = [model_nickname, dataset_id, standardize, hpo_metric, hpo_seed, n_agents, agent_i]
df_missing = get_missing_entries(df_runs_parents, columns_names, should_contain_values)
df_missing

# %%
# Join df_runs_raw_parents into df_missing using non_duplicate_columns to get the EXCEPTION column
df_missing_with_exception = df_missing.merge(
    df_runs_raw_parents[non_duplicate_columns + ["raised_exception", "EXCEPTION"]],
    how="left",
    left_on=["model", "dataset_id", "standardize", "hpo_metric", "hpo_seed"],
    right_on=["model", "dataset_id", "standardize", "hpo_metric", "hpo_seed"],
)
df_missing_with_exception[["model", "dataset_id", "standardize", "hpo_metric", "hpo_seed", "raised_exception", "EXCEPTION"]]

# %%
df_missing_dict = df_missing.copy()
# get only rows from high_mem_tuples
# df_missing_dict = df_missing_dict.merge(high_mem_tuples, on=["model", "dataset_id"], how="left", indicator=True)
# df_missing_dict = df_missing_dict[df_missing_dict["_merge"] == "both"].drop(columns="_merge")
# exclude rows that are in missing_ari_tuples
# df_missing_dict = df_missing_dict.merge(
# 	missing_ari_tuples, on=["model", "dataset_id"], how="left", indicator=True
# )
# df_missing_dict = df_missing_dict[df_missing_dict["_merge"] == "left_only"].drop(columns="_merge")
# exclude rows that are in high_mem_tuples
# df_missing_dict = df_missing_dict.merge(
# 	high_mem_tuples, on=["model", "dataset_id"], how="left", indicator=True
# )
# df_missing_dict = df_missing_dict[df_missing_dict["_merge"] == "left_only"].drop(columns="_merge")
# to_drop = pd.concat([missing_ari_tuples, high_mem_tuples], ignore_index=True)
# df_missing_dict = df_missing_dict[df_missing_dict["_merge"] == "left_only"].drop(columns="_merge")

# %%
# get rid of -60
df_missing_dict["model"] = df_missing_dict["model"].str.replace("-50", "")
df_missing_dict["seed_dataset_order"] = df_missing_dict["hpo_seed"]

# %%
df_missing_dict

# %%
missing_dict = {}
for model in df_missing_dict["model"].unique():
    sub = df_missing_dict[df_missing_dict["model"] == model].drop(columns=["model"])
    # standardize = True
    sub_standardized = sub.loc[sub["standardize"] == True].copy()
    sub_standardized["standardize"] = ''
    sub_standardized_dict = sub_standardized.to_dict(orient="records")
    # standardize = False
    sub_not_standardized = sub.loc[sub["standardize"] == False].copy()
    sub_not_standardized.drop(columns=["standardize"], inplace=True)
    sub_not_standardized_dict = sub_not_standardized.to_dict(orient="records")
    # combine both dictionaries
    missing_dict[model] = sub_standardized_dict + sub_not_standardized_dict
if len(missing_dict) != 0:
    with open(results_dir / 'missing_dict.pkl', 'wb') as f:
        pickle.dump(missing_dict, f)

# %%
missing_dict


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
# ## Some Models

# %%
df_runs_parents['model'].unique()

# %%
model_names = {
    "KMeans-50": "LocalKMeans",
    "VPC-50": "VPC",
    "VeCoHiRF-50": "VeCoHiRF",
    "VeCoHiRF-1iter-50": "VeCoHiRF-1iter",
    "VeCoHiRF-1iter-1iter-50": "VeCoHiRF-1iter-1iter",
	"VeCoHiRF-DBSCAN-50": "VeCoHiRF-DBSCAN",
    "VeCoHiRF-KernelRBF-50": "VeCoHiRF-KernelRBF",
    "VeCoHiRF-SC-SRGF-50": "VeCoHiRF-SC-SRGF",
    # "DistributedKMeansLocal-50": "DistributedKMeansLocal",
    "DistributedKMeans-50": "DistributedKMeans",
    "V2way-50": "V2way",
    "CoresetKMeans-50": "CoresetKMeans",
}

dataset_names = {
    "binary_alpha_digits": "binary-alpha-digits",
	"mnist_784": "mnist",
}  # otherwise we get an error in latex

# Filter to only standardized runs
df = df_runs_parents.copy()
df = df.loc[df['standardize'] == True]
df = df.loc[df['model'].isin(model_names.keys())]
df = df.loc[df['dataset_id'] != 40685]
df = df.replace({"model": model_names})
df = df.replace({"dataset_name": dataset_names})

# Filter to only runs with hpo_seed in range(5)
df = df.loc[df['hpo_seed'].isin(range(5))]

# %%
hpo_metrics = [
    "adjusted_rand_mean",
    # "adjusted_mutual_info",
    # "calinski_harabasz_score",
    # "silhouette",
    # "davies_bouldin_score",
    # "normalized_mutual_info",
]

hpo_metrics_rename = [
    "ARI",
    # "AMI",
    # "Calinski",
    # "Silhouette",
    # "Davies-Bouldin",
    # "NMI",
]

dfs_metrics = {}

for hpo_metric, hpo_metric_rename in zip(hpo_metrics, hpo_metrics_rename):
    df_metric = df.loc[df['hpo_metric'] == hpo_metric][
        ['dataset_name', 'model', 'hpo_seed', f'best/{hpo_metric}']
    ].rename(columns={f'best/{hpo_metric}': hpo_metric_rename})
    df_metric = df_metric.dropna(subset=[hpo_metric_rename])
    df_metric = df_metric.set_index(['dataset_name', 'model', 'hpo_seed'])
    df_metric = df_metric.astype({hpo_metric_rename: float})
    dfs_metrics[hpo_metric_rename] = df_metric

df_metrics = pd.concat(dfs_metrics.values(), axis=1, join="outer")
df_metrics = df_metrics.reset_index()

# calculate mean and std
df_metrics = df_metrics.groupby(['dataset_name', 'model']).agg(['mean', 'std'])
# flatten multiindex columns
df_metrics.columns = [' '.join(col).strip() for col in df_metrics.columns.values]
# drop hpo_seed level
df_metrics = df_metrics.drop(columns=['hpo_seed mean', 'hpo_seed std'])
# Rename index levels
df_metrics.index.names = ["Dataset", "Model"]
# df_metrics["Davies-Bouldin"] = df_metrics["Davies-Bouldin"].astype(float)
# create columns Metric (Mean ± Std)
# for metric in hpo_metrics_rename:
#     df_metrics[f"{metric}"] = df_metrics[f"{metric} mean"].round(3).astype(str) + " $\\pm$ " + df_metrics[f"{metric} std"].round(3).astype(str)

for metric in hpo_metrics_rename:
    df_metrics[f"{metric}"] = (
        df_metrics[f"{metric} mean"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
        + " $\\pm$ "
        + df_metrics[f"{metric} std"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "No Run")
    )


# Reset Seed level
# df_metrics = df_metrics.reset_index(level="Seed")

# %%
df_metrics

# %%
# Add mean time columns to the existing df_metrics dataframe
# Using the same filtering approach as the original df_metrics
df = df_runs_parents.copy()
df = df.loc[df["standardize"] == True]
df = df.loc[df["model"].isin(model_names.keys())]
df = df.replace({"model": model_names})
df = df.replace({"dataset_name": dataset_names})
df = df.loc[df["dataset_id"] != 40685]
# Filter to only runs with hpo_seed in range(5)
df = df.loc[df["hpo_seed"].isin(range(5))]

# Calculate mean and std times for each dataset-model combination across all metrics
df_times = (
    df.groupby(["dataset_name", "model"])
    .agg({"best/elapsed_time": ["mean", "std"], "fit_model_return_elapsed_time": ["mean", "std"]})
    .rename(columns={"best/elapsed_time": "Best Time", "fit_model_return_elapsed_time": "HPO Time"})
)

# Flatten multiindex columns
df_times.columns = [' '.join(col).strip() for col in df_times.columns.values]
# Set the same index structure as df_metrics
df_times.index.names = ["Dataset", "Model"]

df_times["Best Time"] = (
	df_times["Best Time mean"].apply(lambda x: f"{x:4.3f}" if not pd.isna(x) else "No Run")
	+ " $\\pm$ " 
	+ df_times["Best Time std"].apply(lambda x: f"{x:4.3f}" if not pd.isna(x) else "No Run")
)
df_times["HPO Time"] = (
	df_times["HPO Time mean"].apply(lambda x: f"{x:4.3f}" if not pd.isna(x) else "No Run")
	+ " $\\pm$ "
	+ df_times["HPO Time std"].apply(lambda x: f"{x:4.3f}" if not pd.isna(x) else "No Run")
)

# Join with the existing df_metrics (verify we have the same number of rows!)
df_metrics = df_metrics.join(df_times, how="outer")

# %%
# # Create a time-based dataframe with elapsed times for each metric optimization
# # Using the same filtering approach as the original df_metrics
# df_filtered = df_runs_parents.loc[df_runs_parents['standardize'] == True].copy()
# df_filtered = df_filtered.loc[df_filtered['model'].isin(model_names.keys())]
# df_filtered = df_filtered.replace({"model": model_names})
# df_filtered = df_filtered.replace({"dataset_name": dataset_names})

# # Create separate dataframes for each metric optimization with time columns
# df_ari_time = df_filtered.loc[df_filtered['hpo_metric'] == 'adjusted_rand'][
#     ['dataset_name', 'model', 'best/elapsed_time', 'fit_model_return_elapsed_time']
# ].rename(columns={'best/elapsed_time': 'ARI_best_time', 'fit_model_return_elapsed_time': 'ARI_total_time'})

# df_ami_time = df_filtered.loc[df_filtered['hpo_metric'] == 'adjusted_mutual_info'][
#     ['dataset_name', 'model', 'best/elapsed_time', 'fit_model_return_elapsed_time']
# ].rename(columns={'best/elapsed_time': 'AMI_best_time', 'fit_model_return_elapsed_time': 'AMI_total_time'})

# df_nmi_time = df_filtered.loc[df_filtered['hpo_metric'] == 'normalized_mutual_info'][
#     ['dataset_name', 'model', 'best/elapsed_time', 'fit_model_return_elapsed_time']
# ].rename(columns={'best/elapsed_time': 'NMI_best_time', 'fit_model_return_elapsed_time': 'NMI_total_time'})

# df_calinski_time = df_filtered.loc[df_filtered['hpo_metric'] == 'calinski_harabasz_score'][
#     ['dataset_name', 'model', 'best/elapsed_time', 'fit_model_return_elapsed_time']
# ].rename(columns={'best/elapsed_time': 'Calinski_best_time', 'fit_model_return_elapsed_time': 'Calinski_total_time'})

# df_silhouette_time = df_filtered.loc[df_filtered['hpo_metric'] == 'silhouette'][
#     ['dataset_name', 'model', 'best/elapsed_time', 'fit_model_return_elapsed_time']
# ].rename(columns={'best/elapsed_time': 'Silhouette_best_time', 'fit_model_return_elapsed_time': 'Silhouette_total_time'})

# df_davies_bouldin_time = df_filtered.loc[df_filtered['hpo_metric'] == 'davies_bouldin_score'][
#     ['dataset_name', 'model', 'best/elapsed_time', 'fit_model_return_elapsed_time']
# ].rename(columns={'best/elapsed_time': 'Davies-Bouldin_best_time', 'fit_model_return_elapsed_time': 'Davies-Bouldin_total_time'})

# # Remove missing values before setting index
# df_ari_time = df_ari_time.dropna(subset=["ARI_best_time", "ARI_total_time"])
# df_ami_time = df_ami_time.dropna(subset=["AMI_best_time", "AMI_total_time"])
# df_nmi_time = df_nmi_time.dropna(subset=["NMI_best_time", "NMI_total_time"])
# df_calinski_time = df_calinski_time.dropna(subset=["Calinski_best_time", "Calinski_total_time"])
# df_silhouette_time = df_silhouette_time.dropna(subset=["Silhouette_best_time", "Silhouette_total_time"])
# df_davies_bouldin_time = df_davies_bouldin_time.dropna(subset=["Davies-Bouldin_best_time", "Davies-Bouldin_total_time"])

# # Set multi-index for all dataframes
# df_ari_time = df_ari_time.set_index(["dataset_name", "model"])
# df_ami_time = df_ami_time.set_index(["dataset_name", "model"])
# df_nmi_time = df_nmi_time.set_index(["dataset_name", "model"])
# df_calinski_time = df_calinski_time.set_index(["dataset_name", "model"])
# df_silhouette_time = df_silhouette_time.set_index(["dataset_name", "model"])
# df_davies_bouldin_time = df_davies_bouldin_time.set_index(["dataset_name", "model"])

# # Combine all time metrics into a single dataframe using outer join
# df_time_metrics = df_ari_time.join(df_ami_time, how="outer").join(df_nmi, how="outer").join(df_calinski_time, how="outer").join(df_silhouette_time, how="outer").join(df_davies_bouldin_time, how="outer")

# # Rename index levels
# df_time_metrics.index.names = ["Dataset", "Model"]

# %%
df_metrics

# %%
df = df_metrics.copy()
df = df.loc[('alizadeh-2000-v2', 'VeCoHiRF-SC-SRGF')]
df

# %% [markdown]
# The following will provide the latex code for a clean table, we only need to make a little adjustement in the first line to delete the "key" and have only one header. For the longtable environment (full data) we need to add the "\*" at the end of lines we dont want to have a page break. We also should replace the entire begin{table} ... end{table} by begin{longtable} ... end{longtable} in the latex file, if you want to put caption and labels you should break the line after with '\\' (put both on the same line!)
#

# %%
df_latex = df_metrics.copy()
columns_to_hide = [col for col in df_latex.columns if col not in (hpo_metrics_rename + ["Best Time", "HPO Time"])]
# columns_to_hide += ["NMI"]
highlight_max_ari = partial(highlight_max, column_name="ARI mean")
# highlight_max_ami = partial(highlight_max, column_name="AMI mean")
# highlight_max_calinski = partial(highlight_max, column_name="Calinski mean")
# highlight_max_silhouette = partial(highlight_max, column_name="Silhouette mean")
# highlight_min_davies_bouldin = partial(highlight_min, column_name="Davies-Bouldin mean")
highlight_min_best_time = partial(highlight_min, column_name="Best Time mean")
highlight_min_hpo_time = partial(highlight_min, column_name="HPO Time mean")
underline_2nd_max_ari = partial(underline_2nd_max, column_name="ARI mean")
# underline_2nd_max_ami = partial(underline_2nd_max, column_name="AMI mean")
# underline_2nd_max_calinski = partial(underline_2nd_max, column_name="Calinski mean")
# underline_2nd_max_silhouette = partial(underline_2nd_max, column_name="Silhouette mean")
# underline_2nd_min_davies_bouldin = partial(underline_2nd_min, column_name="Davies-Bouldin mean")
underline_2nd_min_best_time = partial(underline_2nd_min, column_name="Best Time mean")
underline_2nd_min_hpo_time = partial(underline_2nd_min, column_name="HPO Time mean")
print(
    df_latex.style.apply(highlight_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(underline_2nd_max_ari, subset=["ARI", "ARI mean"], axis=None)
    # .apply(highlight_max_ami, subset=["AMI", "AMI mean"], axis=None)
    # .apply(underline_2nd_max_ami, subset=["AMI", "AMI mean"], axis=None)
    # .apply(highlight_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    # .apply(underline_2nd_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    # .apply(highlight_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    # .apply(underline_2nd_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    # .apply(highlight_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    # .apply(underline_2nd_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .hide(columns_to_hide, axis=1)
    .to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_latex.columns)), #- len(columns_to_hide)),
        environment="longtable",
    )
)

# %% [markdown]
# # KMeans

# %%
df_latex = df_metrics.copy()
columns_to_hide = [col for col in df_latex.columns if col not in (hpo_metrics_rename + ["Best Time", "HPO Time"])]
columns_to_hide += ["NMI"]
datasets_to_keep = [
    "garber-2001",
    "alizadeh-2000-v2",
    "golub-1999-v2",
    "armstrong-2002-v1",
    "nursery",
    "segment",
]
models_to_keep = [
    "K-Means",
    "CoHiRF",
    "Batch CoHiRF",
]
df_latex = df_latex.loc[
    df_latex.index.get_level_values("Dataset").isin(datasets_to_keep)
    & df_latex.index.get_level_values("Model").isin(models_to_keep),
    :,
]
highlight_max_ari = partial(highlight_max, column_name="ARI mean")
highlight_max_ami = partial(highlight_max, column_name="AMI mean")
highlight_max_calinski = partial(highlight_max, column_name="Calinski mean")
highlight_max_silhouette = partial(highlight_max, column_name="Silhouette mean")
highlight_min_davies_bouldin = partial(highlight_min, column_name="Davies-Bouldin mean")
highlight_min_best_time = partial(highlight_min, column_name="Best Time mean")
highlight_min_hpo_time = partial(highlight_min, column_name="HPO Time mean")
underline_2nd_max_ari = partial(underline_2nd_max, column_name="ARI mean")
underline_2nd_max_ami = partial(underline_2nd_max, column_name="AMI mean")
underline_2nd_max_calinski = partial(underline_2nd_max, column_name="Calinski mean")
underline_2nd_max_silhouette = partial(underline_2nd_max, column_name="Silhouette mean")
underline_2nd_min_davies_bouldin = partial(underline_2nd_min, column_name="Davies-Bouldin mean")
underline_2nd_min_best_time = partial(underline_2nd_min, column_name="Best Time mean")
underline_2nd_min_hpo_time = partial(underline_2nd_min, column_name="HPO Time mean")
print(
    df_latex.style.apply(highlight_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(underline_2nd_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(highlight_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(underline_2nd_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(highlight_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(underline_2nd_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(highlight_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(underline_2nd_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(highlight_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .apply(underline_2nd_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .hide(columns_to_hide, axis=1)
    .to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_latex.columns) - len(columns_to_hide)),
        # environment="longtable",
    )
)

# %% [markdown]
# # Kernel KMeans

# %%
df_latex = df_metrics.copy()
columns_to_hide = [col for col in df_latex.columns if col not in (hpo_metrics_rename + ["Best Time", "HPO Time"])]
columns_to_hide += ["NMI"]
datasets_to_keep = [
    "khan-2001",
    "bittner-2000",
    "iris",
    "satimage",
]
models_to_keep = [
    "Kernel RBF K-Means",
    "CoHiRF-KernelRBF",
    "Batch CoHiRF-KernelRBF",
]
df_latex = df_latex.loc[
    df_latex.index.get_level_values("Dataset").isin(datasets_to_keep)
    & df_latex.index.get_level_values("Model").isin(models_to_keep),
    :,
]
df_latex = df_latex.loc[
    df_latex.index.get_level_values("Dataset").isin(datasets_to_keep)
    & df_latex.index.get_level_values("Model").isin(models_to_keep),
    :,
]
highlight_max_ari = partial(highlight_max, column_name="ARI mean")
highlight_max_ami = partial(highlight_max, column_name="AMI mean")
highlight_max_calinski = partial(highlight_max, column_name="Calinski mean")
highlight_max_silhouette = partial(highlight_max, column_name="Silhouette mean")
highlight_min_davies_bouldin = partial(highlight_min, column_name="Davies-Bouldin mean")
highlight_min_best_time = partial(highlight_min, column_name="Best Time mean")
highlight_min_hpo_time = partial(highlight_min, column_name="HPO Time mean")
underline_2nd_max_ari = partial(underline_2nd_max, column_name="ARI mean")
underline_2nd_max_ami = partial(underline_2nd_max, column_name="AMI mean")
underline_2nd_max_calinski = partial(underline_2nd_max, column_name="Calinski mean")
underline_2nd_max_silhouette = partial(underline_2nd_max, column_name="Silhouette mean")
underline_2nd_min_davies_bouldin = partial(underline_2nd_min, column_name="Davies-Bouldin mean")
underline_2nd_min_best_time = partial(underline_2nd_min, column_name="Best Time mean")
underline_2nd_min_hpo_time = partial(underline_2nd_min, column_name="HPO Time mean")
print(
    df_latex.style.apply(highlight_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(underline_2nd_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(highlight_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(underline_2nd_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(highlight_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(underline_2nd_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(highlight_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(underline_2nd_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(highlight_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .apply(underline_2nd_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .hide(columns_to_hide, axis=1)
    .to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_latex.columns) - len(columns_to_hide)),
        # environment="longtable",
    )
)

# %% [markdown]
# # DBSCAN

# %%
df_latex = df_metrics.copy()
columns_to_hide = [col for col in df_latex.columns if col not in (hpo_metrics_rename + ["Best Time", "HPO Time"])]
columns_to_hide += ["NMI"]
datasets_to_keep = ["ecoli", "binary-alpha-digits", "segment", "chowdary-2006", "shuttle"]
models_to_keep = [
    "DBSCAN",
    "CoHiRF-DBSCAN",
    "Batch CoHiRF-DBSCAN",
]
df_latex = df_latex.loc[
    df_latex.index.get_level_values("Dataset").isin(datasets_to_keep)
    & df_latex.index.get_level_values("Model").isin(models_to_keep),
    :,
]
highlight_max_ari = partial(highlight_max, column_name="ARI mean")
highlight_max_ami = partial(highlight_max, column_name="AMI mean")
highlight_max_calinski = partial(highlight_max, column_name="Calinski mean")
highlight_max_silhouette = partial(highlight_max, column_name="Silhouette mean")
highlight_min_davies_bouldin = partial(highlight_min, column_name="Davies-Bouldin mean")
highlight_min_best_time = partial(highlight_min, column_name="Best Time mean")
highlight_min_hpo_time = partial(highlight_min, column_name="HPO Time mean")
underline_2nd_max_ari = partial(underline_2nd_max, column_name="ARI mean")
underline_2nd_max_ami = partial(underline_2nd_max, column_name="AMI mean")
underline_2nd_max_calinski = partial(underline_2nd_max, column_name="Calinski mean")
underline_2nd_max_silhouette = partial(underline_2nd_max, column_name="Silhouette mean")
underline_2nd_min_davies_bouldin = partial(underline_2nd_min, column_name="Davies-Bouldin mean")
underline_2nd_min_best_time = partial(underline_2nd_min, column_name="Best Time mean")
underline_2nd_min_hpo_time = partial(underline_2nd_min, column_name="HPO Time mean")
print(
    df_latex.style.apply(highlight_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(underline_2nd_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(highlight_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(underline_2nd_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(highlight_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(underline_2nd_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(highlight_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(underline_2nd_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(highlight_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .apply(underline_2nd_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .hide(columns_to_hide, axis=1)
    .to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_latex.columns) - len(columns_to_hide)),
        # environment="longtable",
    )
)

# %% [markdown]
# # SC-SRGF
#

# %%
df_latex = df_metrics.copy()
columns_to_hide = [col for col in df_latex.columns if col not in (hpo_metrics_rename + ["Best Time", "HPO Time"])]
columns_to_hide += ["NMI"]
datasets_to_keep = ["alizadeh-2000-v3", "alizadeh-2000-v2", "har", "satimage", "chowdary-2006"]
models_to_keep = [
    "SC-SRGF",
	"CoHiRF-SC-SRGF",
    "Batch CoHiRF-SC-SRGF",
]
df_latex = df_latex.loc[
    df_latex.index.get_level_values("Dataset").isin(datasets_to_keep)
    & df_latex.index.get_level_values("Model").isin(models_to_keep),
    :,
]
highlight_max_ari = partial(highlight_max, column_name="ARI mean")
highlight_max_ami = partial(highlight_max, column_name="AMI mean")
highlight_max_calinski = partial(highlight_max, column_name="Calinski mean")
highlight_max_silhouette = partial(highlight_max, column_name="Silhouette mean")
highlight_min_davies_bouldin = partial(highlight_min, column_name="Davies-Bouldin mean")
highlight_min_best_time = partial(highlight_min, column_name="Best Time mean")
highlight_min_hpo_time = partial(highlight_min, column_name="HPO Time mean")
underline_2nd_max_ari = partial(underline_2nd_max, column_name="ARI mean")
underline_2nd_max_ami = partial(underline_2nd_max, column_name="AMI mean")
underline_2nd_max_calinski = partial(underline_2nd_max, column_name="Calinski mean")
underline_2nd_max_silhouette = partial(underline_2nd_max, column_name="Silhouette mean")
underline_2nd_min_davies_bouldin = partial(underline_2nd_min, column_name="Davies-Bouldin mean")
underline_2nd_min_best_time = partial(underline_2nd_min, column_name="Best Time mean")
underline_2nd_min_hpo_time = partial(underline_2nd_min, column_name="HPO Time mean")
print(
    df_latex.style.apply(highlight_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(underline_2nd_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(highlight_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(underline_2nd_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(highlight_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(underline_2nd_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(highlight_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(underline_2nd_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(highlight_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .apply(underline_2nd_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .hide(columns_to_hide, axis=1)
    .to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_latex.columns) - len(columns_to_hide)),
        # environment="longtable",
    )
)

# %% [markdown]
# # COIL 20

# %%
df_latex = df_metrics.copy()
columns_to_hide = [col for col in df_latex.columns if col not in (hpo_metrics_rename + ["Best Time", "HPO Time"])]
columns_to_hide += ["NMI"]
datasets_to_keep = ["coil-20", "mnist"]
models_to_keep = [
    "K-Means",
    "CoHiRF",
    "Batch CoHiRF",
    "Kernel RBF K-Means",
    "CoHiRF-KernelRBF",
    "Batch CoHiRF-KernelRBF",
    "DBSCAN",
    "CoHiRF-DBSCAN",
    "Batch CoHiRF-DBSCAN",
    "SC-SRGF",
    "Batch CoHiRF-SC-SRGF",
]
df_latex = df_latex.loc[
    df_latex.index.get_level_values("Dataset").isin(datasets_to_keep)
    & df_latex.index.get_level_values("Model").isin(models_to_keep),
    :,
]
highlight_max_ari = partial(highlight_max, column_name="ARI mean")
highlight_max_ami = partial(highlight_max, column_name="AMI mean")
highlight_max_calinski = partial(highlight_max, column_name="Calinski mean")
highlight_max_silhouette = partial(highlight_max, column_name="Silhouette mean")
highlight_min_davies_bouldin = partial(highlight_min, column_name="Davies-Bouldin mean")
highlight_min_best_time = partial(highlight_min, column_name="Best Time mean")
highlight_min_hpo_time = partial(highlight_min, column_name="HPO Time mean")
underline_2nd_max_ari = partial(underline_2nd_max, column_name="ARI mean")
underline_2nd_max_ami = partial(underline_2nd_max, column_name="AMI mean")
underline_2nd_max_calinski = partial(underline_2nd_max, column_name="Calinski mean")
underline_2nd_max_silhouette = partial(underline_2nd_max, column_name="Silhouette mean")
underline_2nd_min_davies_bouldin = partial(underline_2nd_min, column_name="Davies-Bouldin mean")
underline_2nd_min_best_time = partial(underline_2nd_min, column_name="Best Time mean")
underline_2nd_min_hpo_time = partial(underline_2nd_min, column_name="HPO Time mean")
print(
    df_latex.style.apply(highlight_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(underline_2nd_max_ari, subset=["ARI", "ARI mean"], axis=None)
    .apply(highlight_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(underline_2nd_max_ami, subset=["AMI", "AMI mean"], axis=None)
    .apply(highlight_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(underline_2nd_max_calinski, subset=["Calinski", "Calinski mean"], axis=None)
    .apply(highlight_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(underline_2nd_max_silhouette, subset=["Silhouette", "Silhouette mean"], axis=None)
    .apply(highlight_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .apply(underline_2nd_min_davies_bouldin, subset=["Davies-Bouldin", "Davies-Bouldin mean"], axis=None)
    .hide(columns_to_hide, axis=1)
    .to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        column_format="ll" + "l" * (len(df_latex.columns) - len(columns_to_hide)),
        # environment="longtable",
    )
)

# %% [markdown]
# # Debug and explore

# %%
df = df_runs_raw_parents.copy()

# %%
df = df.loc[df["status"].isin(["FAILED", "SCHEDULED"])]

# %%
df

# %%
runs_to_delete_parents = list(df.index)

# %%
# get children of runs iteratively
runs_to_delete = []
while 1:
    df = df_runs_raw.copy()
    df = df.loc[df["mlflow.parentRunId"].isin(runs_to_delete_parents)]
    runs_to_delete_children = list(df.index)
    runs_to_delete = runs_to_delete + runs_to_delete_children
    runs_to_delete_parents = runs_to_delete_children
    if len(runs_to_delete_parents) == 0:
        break

# %%
run_uuid_query = [f"'{run_id}'" for run_id in runs_to_delete]
run_uuid_query = ', '.join(run_uuid_query)

# %%
print(run_uuid_query)

# %%
# query to update name in experiments from adjusted_rand_mean-VeCoHiRF-1iter to adjusted_rand_mean-VeCoHiRF-full-1iter
query = f"""
UPDATE experiments
SET name = 'adjusted_rand_mean-VeCoHiRF-full-1iter'
WHERE name = 'adjusted_rand_mean-VeCoHiRF-1iter'
"""
with engine.begin() as conn:
    conn.execute(text(query))

# %%
# query to update name in experiments from adjusted_rand_mean-VeCoHiRF-1iter-1iter to adjusted_rand_mean-VeCoHiRF-1iter
query = f"""
UPDATE experiments
SET name = 'adjusted_rand_mean-VeCoHiRF-1iter'
WHERE name = 'adjusted_rand_mean-VeCoHiRF-1iter-1iter'
"""
with engine.begin() as conn:
    conn.execute(text(query))

# %%
# query to update value in params from VeCoHiRF-1iter to VeCoHiRF-full-1iter
query = f"""
UPDATE params
SET value = 'VeCoHiRF-full-1iter'
WHERE value = 'VeCoHiRF-1iter' AND key = 'model_alias'
"""
with engine.begin() as conn:
    conn.execute(text(query))

# %%
# query to update value in params from VeCoHiRF-1iter-1iter to VeCoHiRF-1iter
query = f"""
UPDATE params
SET value = 'VeCoHiRF-1iter'
WHERE value = 'VeCoHiRF-1iter-1iter' AND key = 'model_alias'
"""
with engine.begin() as conn:
    conn.execute(text(query))

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
