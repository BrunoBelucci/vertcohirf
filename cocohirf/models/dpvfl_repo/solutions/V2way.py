import numpy as np
import copy
from sklearn.cluster import KMeans
import itertools
import logging
import pandas as pd

from .VPrivClustering import VPrivClustering
from .solver_factory import solver_mapping
from util.save_results import save_result_to_json
from util.eval_centers import eval_centers, eval_homogeneity_score
from util.volh import volh_perturb, volh_membership, rr_membership, rr_perturb
from util.load_config import generate_local_config
from util.fmsketch import get_one_n_two_way_intersection_est
from util.postprocess import norm_sub
from util.local_k import local_k_choose

'''
This is an implementation of the paper Hu Ding et al. "K-Means Clustering with Distributed Dimensions"
The paper introduces a non-private VFL solution for K-means clustering problem
'''


class V2way(VPrivClustering):
    def __init__(self, config, tag, **kwargs):
        super().__init__(config, tag, )
        self.config = config
        self.tag = tag
        self.k = config['k']
        assert 'eps' in config and 'intersection_method' in config
        self.eps = config['eps']
        self.intersection_method = config['intersection_method']

        if 'local_solver' in config:
            self.local_solver = solver_mapping[config['local_solver']]
        else:
            self.local_solver = solver_mapping['basic']
        self.centers = None
        self.private_intersections = []
        self.clean_intersections = []
        self.clean_centers = None
        self.max_itr = 100
        self.update_step = 0.1
        self.one_ways = None
        self.clean_oneways = None

    def fit(self, data, run_clean: bool = True, true_labels: np.array = None):
        if self.intersection_method in ['fmsketch']:
            n = int(self.config['n'] + np.random.laplace(0, 1 / (0.02 * self.eps)))
            self.eps *= 0.98
        else:
            n = self.config['n']
        logging.info(f'n: {n}')
        centers = []
        clean_memberships = []
        parties = len(data)
        if 'local_cluster_budget' in self.config:
            local_clustering_eps = self.eps * self.config['local_cluster_budget'] / parties
            local_aggregation_eps = self.eps / parties - local_clustering_eps
        elif self.local_solver == solver_mapping['basic']:
            # for experiments: if the non-private k-means algorithm is applied on each party's local data
            # to show the effect of different intersection algorithms
            local_aggregation_eps = self.eps / 2 / parties
            local_clustering_eps = np.inf
        else:
            # current default version use half of the eps for finding local centroids, half for reporting membership
            local_clustering_eps = self.eps / 2 / parties
            local_aggregation_eps = self.eps / parties - local_clustering_eps

        if 'local_k' in self.config:
            if self.config['local_k'] == 'auto':
                delta = 1 / data[0].shape[0]
                local_k = local_k_choose(max_k=10, n=data[0].shape[0],
                                         number_sketches=self.config['m'],
                                         epsilon=local_aggregation_eps, delta=delta)
                # todo: ad hoc for rebuttal
                if self.config['T'] > 4:
                    local_k = min(local_k, self.config['k'])
                logging.info(f"auto set local k as {local_k}")
            elif self.config['local_k'] < np.power(self.config['k'], 1/parties):
                local_k = int(np.ceil(np.power(self.config['k'], 1 / parties)))
            else:
                local_k = self.config['local_k']
        else:
            local_k = self.config['k']

        local_config = generate_local_config(self.config['d'], self.config['n'], local_k, eps=local_clustering_eps)
        for idx, subset_data in enumerate(data):
            # each party local run k-means and get centers
            logging.info(f"--> Working on client {idx} {subset_data.shape}")
            solver = self.local_solver(local_config, self.tag)
            solver.fit(subset_data)
            # -> get centers
            centers.append(list(solver.cluster_centers_))
            # # todo: debug
            # print(f"*** len of party {idx} centers: {len(solver.cluster_centers_)}")
        logging.info("local kmeans finished...")
        # exit()

        logging.info(f"Privacy budget for computing aggregation: {local_aggregation_eps}")

        clean_memberships = []
        for idx, subset_data in enumerate(data):
            membership = self.clean_membership(subset_data, centers[idx])
            clean_memberships.append(membership)

        grids, intersection_counts, clean_intersection_counts = self.build_weighted_grids(n,
                                                                                          data,
                                                                                          centers,
                                                                                          clean_memberships,
                                                                                          local_aggregation_eps,
                                                                                          local_k,
                                                                                          run_clean)

        logging.info(f"# of grid nodes: {len(grids)}; # of intersections: {len(intersection_counts)}")
        logging.info(f"intersection sizes: {intersection_counts}")

        if 'normalize' in self.config and self.config['normalize']:
            logging.info(f"postprocessing...")
            intersection_counts = norm_sub(intersection_counts, n=n)

        if self.intersection_method == 'random':
            chosen_idxs = np.random.choice(len(grids), size=self.k).astype(int)
            self.centers = np.array(grids)[chosen_idxs]
            loss = eval_centers(data, self.centers)
        else:
            # run k-means again on the weighted centers
            final_solver = KMeans(n_clusters=self.k, random_state=0)
            # print(grids)
            final_solver.fit(grids, sample_weight=np.array(intersection_counts) + 1e-5)

            loss = eval_centers(data, final_solver.cluster_centers_)
            self.centers = final_solver.cluster_centers_
        self.private_intersections = intersection_counts
        losses = {"private_final_loss": loss}
        losses['local_k'] = int(local_k)

        if run_clean:
            # run k-means again on the weighted centers
            clean_final_solver = KMeans(n_clusters=self.k, random_state=0)
            clean_final_solver.fit(grids, sample_weight=np.array(clean_intersection_counts) + 1e-5)
            clean_score = eval_centers(data, clean_final_solver.cluster_centers_)
            losses["clean_final_loss"] = clean_score
            self.clean_centers = clean_final_solver.cluster_centers_
            logging.info(f"intersection diff {intersection_counts - clean_intersection_counts}")
            if 'label_score' in self.config:
                # print(self.config['label_score'], self.config['label_score'] == True)
                losses['homogeneity'], losses['completeness'] = eval_homogeneity_score(data, self.centers, true_labels)
        self.save_results(losses)

        return self.centers

    def build_weighted_grids(self, n, data, centers, clean_memberships, eps, local_k, run_clean):
        memberships = []
        parties = self.config['T']
        # cartesian product of the local centers
        logging.info("generate cartesian product of local centers ...")
        cartesian = list(itertools.product(*centers))
        grids = []
        for combine in cartesian:
            # grids.append(np.array(list(combine)).flatten())
            grids.append(np.concatenate(list(combine)))

        one_ways = []
        if self.intersection_method in ['ldp', '1way_ldp']:
            # generate private memberships with grr/olh
            for idx, subset_data in enumerate(data):
                # get randomized membership
                client_membership = self.ldp_membership(subset_data, centers[idx], eps)
                memberships.append(client_membership)
                # # todo: debug
                # print(f"*** client membership size {len(client_membership)}")

                # generate one way histogram, and then compute the unbiased
                one_ways.append(self.generate_one_way_from_membership(client_membership, n, eps))

            # generate C(parties, 2) intersection counts， which can be considered as 2-way marginals
            two_ways = {}
            for i in range(parties):
                for j in range(i + 1, parties):
                    # intersection of i-th and j-th cleint memberships
                    i_j_intersections = self.intersection([memberships[i], memberships[j]])
                    i_j_intersection_counts = np.array([len(s) for s in i_j_intersections])
                    # # todo: debug
                    # print(f"******* {i}~{j}: {i_j_intersection_counts}")
                    # make those intersection counts unbiased
                    two_way = self.ldp_intersection_count_adjust(i_j_intersection_counts, eps,
                                                                 parties=2, local_k=local_k)
                    two_way = norm_sub(two_way, n=n)
                    # # enforcing consistency with one ways
                    # two_way = self.two_way_consistency(i, j, two_way, one_ways)
                    two_ways[(i, j)] = two_way
                    logging.info(f"build {i} x {j} intersection 2-way, total count: {np.sum(two_ways[(i, j)])}")
        elif self.intersection_method in ['fmsketch', '1way_fm']:
            priv_config = {'eps': eps, 'delta': 1 / n}
            splits = []
            for idx, membership in enumerate(clean_memberships):
                tmp = [np.array(list(m)) for m in membership]
                splits.append(tmp)
            logging.info(f"compute intersection CA with FM sketch...")
            one_ways, two_ways = get_one_n_two_way_intersection_est(n, splits,
                                                                    m=self.config['m'],
                                                                    gamma=1,
                                                                    priv_config=priv_config,
                                                                    multithreads=20)
            for key, marginal in two_ways.items():
                two_ways[key] = norm_sub(marginal, n=n)
            for i in range(len(one_ways)):
                one_ways[i] = norm_sub(one_ways[i], n=n)
        else:
            raise NotImplementedError
        # all one-ways and two-ways are post-processed by normsub
        self.one_ways = one_ways

        # initialize with uniform
        portions = []
        for hist in one_ways:
            hist *= (n / np.sum(hist))
            portions.append(list(hist / n))
            print(portions[-1])
        portions_combines = list(itertools.product(*portions))
        grid_weights = np.array([n * np.product(p) for p in portions_combines])
        # change grid_weight to pd.Dataframe for easier groupby operation
        grid_weights_df = self.prepare_update(grid_weights, [len(c) for c in centers])
        if not self.intersection_method.startswith('1way'):
            print(grid_weights_df)
            print("one ways:", one_ways)
            # if we just use 1way, then just relies on the dependent assumption, otherwise
            # iteratively update so that the grid weights approach the two-way intersection counts
            self.max_itr = parties * (parties - 1) * 20
            step_size = 0.5
            # todo: ad hoc for rebuttal
            if parties > 4:
                step_size = 0.5
                self.max_itr = parties * (parties - 1) * 15
            for itr in range(self.max_itr):
                print(f"iteration: {itr} {step_size}")
                grid_weights_df = self.one_itr_update(two_ways=two_ways,
                                                      grid_weights_df=grid_weights_df,
                                                      parties=parties,
                                                      step_size=step_size)
                if parties > 4:
                    if step_size < 0.001:
                        continue
                    elif itr % 50 == 0 and itr > 0:
                        step_size *= 0.95
                elif parties == 4:
                    if step_size < 0.001:
                        continue
                    elif itr % 10 == 0 and itr > 0:
                        step_size *= 0.95
                else:
                    step_size *= 0.96
            # exit()

        if run_clean:
            self.clean_oneways = []
            for membership in clean_memberships:
                self.clean_oneways.append([len(m) for m in membership])
            clean_intersections = self.intersection(clean_memberships)
            clean_intersection_counts = np.array([len(s) for s in clean_intersections])
            logging.info(f"(clean) intersection sizes: {clean_intersection_counts}")
            self.clean_intersections = clean_intersection_counts
            return grids, grid_weights_df.values.flatten(), clean_intersection_counts
        else:
            return grids, grid_weights_df.values.flatten(), []

    def save_results(self, losses):
        if self.config['T'] > 4:
            results = {
                "config": self.config,
                "losses": losses,
                "final_centers": self.centers,
            }
        else:
            results = {
                "config": self.config,
                "losses": losses,
                "final_centers": self.centers,
                "private_intersections": self.private_intersections,
                "clean_intersections": self.clean_intersections,
                "clean_centers": self.clean_centers,
                "one-ways": self.one_ways,
                "clean-ways": self.clean_oneways,
            }
        save_result_to_json(results, self.tag, experiment=self.config['dataset'] + '-' \
                                                          + self.config['intersection_method'] + "-V2way")

    def generate_one_way_from_membership(self, membership, n, eps):
        local_k = len(membership)
        one_way = np.array([len(m) for m in membership])
        if self.k > 3 * int(round(np.exp(eps))) + 2:
            g = int(round(np.exp(eps))) + 1
            p = np.exp(eps) / (np.exp(eps) + g - 1)
            q = 1.0 / (np.exp(eps) + g - 1)
            debiased = (one_way - n / g) / (p - 1 / g)
        else:
            p = np.exp(eps) / (np.exp(eps) + local_k - 1)
            q = 1.0 / (np.exp(eps) + local_k - 1)
            debiased = (one_way - n * q) / (p - q)
        debiased = norm_sub(debiased, n=n)
        return debiased

    def generate_idex_mappings(self, parties, local_ks):
        full = [list(range(local_k)) for local_k in local_ks]
        cartesian = list(itertools.product(*full))
        cartesian2idx = {}
        for idx, c in enumerate(cartesian):
            cartesian2idx[c] = idx
        return cartesian, cartesian2idx

    def prepare_update(self, grid_weights, local_ks):
        full = [list(range(local_k)) for local_k in local_ks]
        cartesian = list(itertools.product(*full))
        index = pd.MultiIndex.from_tuples(cartesian, names=list(range(len(local_ks))))
        df = pd.DataFrame(grid_weights, index=index)
        return df

    def two_way_consistency(self, i: int, j: int, two_way: np.array, one_ways: list):
        i, j = min(i, j), max(i, j)
        centers = [list(range(len(one_ways[i]))), list(range(len(one_ways[j])))]
        cartesian = list(itertools.product(*centers))
        index = pd.MultiIndex.from_tuples(cartesian, names=[i, j])
        df = pd.DataFrame(two_way, index=index)
        two_way = df.unstack(level=-1)

        for itr in range(int(self.max_itr / 2)):
            row_sum = np.sum(two_way, axis=1).values
            row_diff = one_ways[i] - row_sum
            two_way += row_diff.reshape((len(row_diff), 1)) * self.update_step
            column_sum = np.sum(two_way, axis=0)
            column_diff = one_ways[j] - column_sum.values
            two_way += column_diff.reshape((1, len(column_diff))) * self.update_step
            logging.info(f"enforcing {i} x {j} 2-way consistency, {np.sum(row_diff)} {np.sum(column_diff)}")
        return two_way.stack().values.flatten()

    def one_itr_update(self, two_ways: dict, grid_weights_df: pd.DataFrame, parties: int, step_size: float):
        i = np.random.choice(parties)
        j = np.random.choice(parties)
        while i == j:
            j = np.random.choice(parties)
        i, j = min(i, j), max(i, j)
        # print(f"update {i, j}")

        # print(grid_weights_df)
        # print(np.sum(grid_weights_df.values), np.max(grid_weights_df.values), np.min(grid_weights_df.values))
        # print(grid_weights_df.groupby(level=2).sum())
        # exit()
        target_marginal = two_ways[(i, j)]
        # pandas groupby to get a two-way
        groupby = grid_weights_df.groupby(level=[i, j])
        cur_marginal = groupby.sum()
        cur_marginal[cur_marginal <= 0] = 1e-5
        diff = target_marginal.reshape(cur_marginal.shape) - cur_marginal
        factor = target_marginal.reshape(cur_marginal.shape) / cur_marginal
        # print("xxx", factor)
        # print(target_marginal.reshape(cur_marginal.shape))
        # print(cur_marginal)
        # exit()

        # update the marginal
        for pair, full in groupby.groups.items():
            # grid_weights_df.loc[full] *= factor.loc[pair]
            if parties >= 4:
                factor = 1 / len(full)
            else:
                factor = 1
            grid_weights_df.loc[full] += diff.loc[pair] * step_size * factor
            grid_weights_df[grid_weights_df <= 0] = 1e-5
        logging.info(f"{i} x {j} loss: {np.linalg.norm(diff.values, 1)}, "
                     f" total count: {np.sum(grid_weights_df.values)}"
                     f" max: {np.max(grid_weights_df.values)}, max idx: {np.argmax(grid_weights_df.values)}"
                     f" min: {np.min(grid_weights_df.values)}")

        return grid_weights_df
