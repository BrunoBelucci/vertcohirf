import numpy as np
from matplotlib import pyplot as plt
import logging
import sys
import os
import pickle
import pandas as pd
import random

'''
    Here generate the data point and ensure that each dimension is bounded to [-r, r]
'''

selected_attrs = [
                  'EXT_SOURCE_2',
                  'OWN_CAR_AGE',
                  'DAYS_REGISTRATION',
                  'DAYS_BIRTH',
                  'AMT_ANNUITY',
                  'AMT_CREDIT',
                  'AMT_REQ_CREDIT_BUREAU_YEAR',
                  'AMT_INCOME_TOTAL',
                  'EXT_SOURCE_1',
                  'DAYS_EMPLOYED',
                  'REGION_POPULATION_RELATIVE',
                  'DAYS_LAST_PHONE_CHANGE',
                  'FLAG_DOCUMENT_3',
                  'DEF_30_CNT_SOCIAL_CIRCLE',
                  'DEF_60_CNT_SOCIAL_CIRCLE',
                  'TOTALAREA_MODE',
                  'HOUR_APPR_PROCESS_START',
                  'COMMONAREA_MEDI',
                  'EXT_SOURCE_3',
                  'DAYS_ID_PUBLISH',
                  'NONLIVINGAPARTMENTS_AVG',
                  ]

# selected_attrs = [
#     'EXT_SOURCE_2',
#     'EXT_SOURCE_1',
#     'EXT_SOURCE_3',
#     'OWN_CAR_AGE',
#     'DAYS_BIRTH',
#     'AMT_ANNUITY',
#     'AMT_CREDIT',
#     'AMT_REQ_CREDIT_BUREAU_YEAR',
#     'AMT_INCOME_TOTAL',
#     'DAYS_EMPLOYED',
#     'DAYS_ID_PUBLISH',
#     'NONLIVINGAPARTMENTS_AVG',
# ]
# 03-05 try, uniform still to good
selected_attrs = [
    'AMT_CREDIT',
    'CNT_FAM_MEMBERS',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'DAYS_REGISTRATION',
    'OBS_30_CNT_SOCIAL_CIRCLE',
    'DAYS_BIRTH',
    'AMT_GOODS_PRICE',
    'DEF_60_CNT_SOCIAL_CIRCLE',
    'CNT_CHILDREN',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'AMT_ANNUITY',
    'YEARS_BUILD_AVG',
]
# 03-05-2 try
selected_attrs = [
    'AMT_CREDIT',
    'CNT_FAM_MEMBERS',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    # 'DAYS_REGISTRATION',
    'OBS_30_CNT_SOCIAL_CIRCLE',
    # 'DAYS_BIRTH',
    'AMT_GOODS_PRICE',
    'DEF_60_CNT_SOCIAL_CIRCLE',
    # 'CNT_CHILDREN',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'AMT_ANNUITY',
    # 'YEARS_BUILD_AVG',
]

selected_attrs = [
    'AMT_CREDIT',
    'CNT_FAM_MEMBERS',
    'ENTRANCES_AVG',
    # 'DEF_30_CNT_SOCIAL_CIRCLE',
    'BASEMENTAREA_AVG',
    'APARTMENTS_AVG',
    'OBS_30_CNT_SOCIAL_CIRCLE',
    'AMT_GOODS_PRICE',
    'FLOORSMAX_AVG',
    # 'DEF_60_CNT_SOCIAL_CIRCLE',
    'FLOORSMIN_MEDI',
    'LIVINGAREA_AVG',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'CNT_CHILDREN',
    'LIVINGAPARTMENTS_AVG',
    'AMT_ANNUITY',
    'ELEVATORS_AVG',
    'COMMONAREA_AVG',
]


class LoanLoader:
    def __init__(self, n, T, **kwargs):
        self.n = n
        self.d = 0
        assert T in [1, 2, 4, 8]
        self.parties = T
        logging.basicConfig(format='%(asctime)s - %(module)s -  %(message)s', level=logging.INFO)
        pass

    def load_data(self):
        raw_filepath = f"./data/loan.csv"
        preprocessed_filepath = f"./data/preprpcessed_loan3_n={self.n}_S={self.parties}.pkl"
        # if os.path.exists(preprocessed_filepath):
        #     logging.info(f"{preprocessed_filepath} exists, load the data file...")
        #     file = open(preprocessed_filepath, 'rb')
        #     dump_data = pickle.load(file)
        #     logging.info(f"Column partitions info:{dump_data['column_partition']}")
        #     self.n = dump_data['data'][0].shape[0]
        #     for s in dump_data['column_partition']:
        #         self.d += len(s)
        #     return dump_data['data']

        logging.info("reading loan datasets...")
        df = pd.read_csv(raw_filepath, nrows=self.n)

        df = df[selected_attrs]
        # df = df.select_dtypes(include=[np.float, np.int])
        # logging.info(f"after filter NA, loaded loan datasets with size {df.shape}...")
        # df['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
        # df['DAYS_EMPLOYED'].replace({365243: 0}, inplace=True)
        df.fillna(df.mean(), inplace=True)
        # df['AMT_INCOME_TOTAL'].clip(upper=1000000, inplace=True)

        for col in df:
            print(df[col].describe())
            # normalize each attr to [-1, 1]
            df[col].clip(upper=df[col].quantile(q=0.95), inplace=True)
            df[col] -= (df[col].min() + df[col].max()) / 2
            df[col] /= df[col].max() + 1e-5
            print("after:", df[col].min(), df[col].max(), df[col].mean())
            print(df[col].describe())
            print("="*20)
        logging.info(f"keep loan datasets with size {df.shape}...")

        # todo: remove select attribute code
        # corr = df.corr()
        # print(corr)
        # for pair in corr[corr >= 0.5].stack().index.tolist():
        #     if pair[0] != pair[1] and ('MODE' not in pair[0]) and ('FLAG' not in pair[0]) and ('MODE' not in pair[1]) and ('FLAG' not in pair[1]):
        #         print(pair, corr.loc[pair[0], pair[1]])
        # exit()

        splited_data, split_group_cols = self.data_split(df)
        dump_data = {"column_partition": split_group_cols, "data": splited_data}
        file = open(preprocessed_filepath, 'wb')
        pickle.dump(dump_data, file)
        logging.info(f"Column partitions info:{dump_data['column_partition']}")
        self.n = dump_data['data'][0].shape[0]
        for s in dump_data['column_partition']:
            self.d += len(s)
        return splited_data

    def data_split(self, df):
        # random.shuffle(selected_attrs)
        m = len(selected_attrs)
        splited_data = []
        if self.parties == 1:
            split_group_cols = [
                selected_attrs
            ]
        elif self.parties == 2:
            split_group_cols = [
                selected_attrs[:int(m / 2)], selected_attrs[int(m / 2):]
            ]
        elif self.parties == 4:
            step = int(m / 4)
            split_group_cols = [
                selected_attrs[i * step: (i + 1) * step] for i in range(4)
            ]
        elif self.parties == 8:
            step = int(m / 8)
            split_group_cols = [
                selected_attrs[i * step: (i + 1) * step] for i in range(8)
            ]
        else:
            raise NotImplementedError

        for cols in split_group_cols:
            splited_data.append(df[cols].values)

        return splited_data, split_group_cols