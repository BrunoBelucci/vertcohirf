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
class TaxiLoader:
    def __init__(self, n, T, **kwargs):
        self.n = n
        assert T in [1, 2, 4]
        self.parties = T
        logging.basicConfig(format='%(asctime)s - %(module)s -  %(message)s', level=logging.INFO)
        pass

    def load_data(self):
        raw_filepath = f"./data/taxi.csv"
        preprocessed_filepath = f"./data/preprpcessed_taxi_n={self.n}_S={self.parties}.pkl"
        if os.path.exists(preprocessed_filepath):
            logging.info(f"{preprocessed_filepath} exists, load the data file...")
            file = open(preprocessed_filepath, 'rb')
            return pickle.load(file)

        time_attrs = ['pickup_datetime', 'dropoff_datetime']
        num_attrs = ['passenger_count', 'pickup_longitude', 'pickup_latitude',  'dropoff_longitude',
                       'dropoff_latitude', 'trip_duration']
        logging.info("reading New York taxi datasets...")
        df = pd.read_csv(raw_filepath, nrows=self.n+500)
        df = df[df['trip_duration'] < 5000]
        df = df.head(n=self.n)
        logging.info(f"loaded New York taxi datasets with size {df.shape}...")
        # print(df.head())
        for col in df:
            # print(col, df[col].dtype)
            if col in time_attrs:
                df[col] = pd.to_datetime(df[col])
                # print("before:", df[col].min(), df[col].max(), df[col].mean())
                df[col] = df[col].astype(int) / 1e9
                # only consider the time in a week
                df[col] = df[col].mod(60 * 60 * 24 * 7)
            if col in num_attrs or col in time_attrs:
                # normalize each attr to [-1, 1]
                df[col] -= (df[col].min() + df[col].max()) / 2
                df[col] /= df[col].max()
                # print("after:", df[col].min(), df[col].max(), df[col].mean())
        splited_data = self.data_split(df)
        file = open(preprocessed_filepath, 'wb')
        pickle.dump(splited_data, file)
        return splited_data

    def data_split(self, df):
        splited_data = []
        if self.parties == 1:
            split_group_cols = [
                ['pickup_datetime', 'dropoff_datetime', 'passenger_count', 'trip_duration',
                 'pickup_longitude', 'pickup_latitude',  'dropoff_longitude','dropoff_latitude']
            ]
        elif self.parties == 2:
            split_group_cols = [
                ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'passenger_count', ],
                ['dropoff_datetime', 'dropoff_longitude', 'dropoff_latitude', 'trip_duration']
            ]
        elif self.parties == 4:
            split_group_cols = [
                [ 'dropoff_longitude', 'pickup_longitude'], ['passenger_count', 'dropoff_datetime'],
                ['pickup_latitude', 'dropoff_latitude'], ['pickup_datetime', 'trip_duration']
            ]
        else:
            raise NotImplementedError

        for cols in split_group_cols:
            splited_data.append(df[cols].values)

        return splited_data