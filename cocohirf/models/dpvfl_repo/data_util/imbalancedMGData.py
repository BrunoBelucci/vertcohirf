import numpy as np
from matplotlib import pyplot as plt
import logging
import sys
import os
import pickle

'''
    Here generate the data point and ensure that each dimension is bounded to [-r, r]
'''
class ImbalancedMixGaussGenerator:
    def __init__(self, k, d, n, r, **kwargs):
        self.k = k
        self.d = d
        self.n = n
        self.r = r
        logging.basicConfig(format='%(asctime)s - %(module)s -  %(message)s', level=logging.INFO)
        self.labels = None

    def load_data(self, cube_r: float = 1.0, plot: bool = False):
        filepath = f"./data/imbalanced_mix_gaussian3_k={self.k}_d={self.d}_n={self.n}.pkl"
        if os.path.exists(filepath):
            logging.info(f"{filepath} exists, load the data file...")
            file = open(filepath, 'rb')
            return pickle.load(file)

        # generate centers
        logging.info(f"paramerters: {self.k, self.d, self.r, self.n}")
        data_centers = np.random.uniform(low=-cube_r, high=cube_r, size=(self.k, self.d))
        for i in range(self.k - 1):
            distances = np.linalg.norm(np.delete(data_centers, i, axis=0) - data_centers[i], ord=2, axis=1)
            while np.min(distances) < self.r * 5:
                data_centers[i] = np.random.uniform(low=-cube_r, high=cube_r, size=(self.k))
                distances = np.linalg.norm(np.delete(data_centers, i, axis=0) - data_centers[i], ord=2, axis=1)
        logging.info(f"Generate centers: {data_centers}")
        logging.info(f"Centers norms:{np.linalg.norm(data_centers, 2, axis=1)}")

        # each cluster sizes
        cluster_sizes = np.ones(shape=self.k)*0.8 + np.random.uniform(size=self.k)
        cluster_sizes = cluster_sizes / np.sum(cluster_sizes) * self.n
        cluster_sizes = cluster_sizes.astype(int)
        cluster_sizes[-1] = self.n - np.sum(cluster_sizes[:-1])
        logging.info(f"generated cluster sizes: {cluster_sizes}, total:{np.sum(cluster_sizes)}")

        # generate data
        data = np.concatenate([np.random.multivariate_normal(
            mean=c,
            cov=self.r * np.eye(self.d),
            size=size, ) for c, size in zip(data_centers, cluster_sizes)],
            axis=0)
        labels = np.concatenate([np.ones(s) * i for i, s in enumerate(cluster_sizes)])
        # make sure all data are in the ball
        data[data > cube_r] = cube_r
        data[data < -cube_r] = -cube_r
        l1_norms = np.linalg.norm(data, 1, axis=1)
        l2_norms = np.linalg.norm(data, 2, axis=1)
        logging.info(f"generating data...")
        logging.info(f"data ranges: max/min value for dimensions: {np.max(data)} {np.min(data)}")
        logging.info(f"data ranges: max l1/l2 norm for dimensions: {np.max(l1_norms)} {np.min(l2_norms)}")
        logging.info(f"labels counts: {np.unique(labels, return_counts=True)}, cluster sizes: {cluster_sizes}")

        if plot:
            plt.plot(data[:, 0], data[:, 1], ".")
            plt.plot(data_centers[:, 0], data_centers[:, 1], ".")
            plt.ylim((-1, 1))
            plt.xlim((-1, 1))
            plt.show()
            exit()
        idx = np.random.permutation(range(len(labels)))
        # shuffle together
        data, labels = data[idx, :], labels[idx]
        self.labels = labels
        # np.random.shuffle(data)

        file = open(filepath, 'wb')
        pickle.dump(data, file)
        return data
