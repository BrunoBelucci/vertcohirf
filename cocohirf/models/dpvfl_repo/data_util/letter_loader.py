import numpy as np
from matplotlib import pyplot as plt
import logging
import sys
import os
import pickle
import pandas as pd

'''
    Here generate the data point and ensure that each dimension is bounded to [-r, r]
'''
class LetterLoader:
    def __init__(self, n, T, **kwargs):
        self.n = 20000
        self.d = 16
        assert T in [1, 2, 4]
        self.parties = T
        logging.basicConfig(format='%(asctime)s - %(module)s -  %(message)s', level=logging.INFO)
        pass

    def load_data(self):
        raw_filepath = f"./data/letter-recognition.data"
        preprocessed_filepath = f"./data/preprpcessed_letter_n={self.n}_S={self.parties}.pkl"
        # if os.path.exists(preprocessed_filepath):
        #     logging.info(f"{preprocessed_filepath} exists, load the data file...")
        #     file = open(preprocessed_filepath, 'rb')
        #     return pickle.load(file)

        logging.info("reading letter-recognition.data datasets...")
        df = pd.read_csv(raw_filepath, nrows=self.n, sep=',', header=None)
        logging.info(f"loaded letter-recognition datasets with size {df.shape}...")
        # print(df.head())
        df.drop(columns=[0], inplace=True)
        print(df.columns)
        for col in df:
            df[col] -= (df[col].min() + df[col].max()) / 2
            df[col] /= df[col].max()
            # print(df[col].describe())
            # print("==="*10)
            # print("after:", df[col].min(), df[col].max(), df[col].mean())
        # corr = df.corr()
        # print(np.mean(np.abs(corr.values)))
        # # sorted_map = {}
        # for pair in corr[(corr >= 0.3) | (corr <= -0.3) ].stack().index.tolist():
        #     if pair[0] != pair[1]:
        #         print(pair, corr.loc[pair[0], pair[1]])
        # exit()

        splited_data = self.data_split(df)
        # file = open(preprocessed_filepath, 'wb')
        # pickle.dump(splited_data, file)
        return splited_data

    def data_split(self, df: pd.DataFrame):
        splited_data = []
        if self.parties == 1:
            split_group_cols = [
                [i for i in range(1, 17)]
            ]
        elif self.parties == 2:
            split_group_cols = [
                # [i * 2 + 1 for i in range(8)],
                # [i * 2 + 2 for i in range(8)],
                [1, 3, 6, 7, 9, 12, 14, 15],
                [2, 4, 5, 8, 10, 11, 13, 16],
            ]
        elif self.parties == 4:
            step = 4
            split_group_cols = [
                # [i*4 + 1 for i in range(0, step)],
                # [i*4 + 2 for i in range(0, step)],
                # [i*4 + 3 for i in range(0, step)],
                # [i*4 + 4 for i in range(0, step)],
                [1, 6, 9, 14],
                [2, 5, 10, 16],
                [3, 7, 12, 15],
                [4, 8, 11, 13],
            ]
        else:
            raise NotImplementedError

        for cols in split_group_cols:
            splited_data.append(df[cols].values)

        logging.info(f"splitted data shape: {[s.shape for s in splited_data]}")

        return splited_data