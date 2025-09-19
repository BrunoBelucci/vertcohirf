import logging
import numpy as np

from .imbalancedMGData import ImbalancedMixGaussGenerator
from .taxi_loader import TaxiLoader
from .loan_loader import LoanLoader
from .letter_loader import LetterLoader


class Dataloader:
    def __init__(self):
        self.k = None
        self.d = None
        self.n = None
        self.r = None
        self.logger = logging.getLogger('Dataloader')
        pass

    def load_data(self, config):
        assert "dataset" in config
        if config["dataset"] == 'imbalancedMG':
            assert "k" in config and "d" in config and "n" in config and "r" in config
            self.k, self.d, self.n, self.r = config["k"], config["d"], config["n"], config["r"]
            # MixGaussGenerator2 generates data where each dimension is bounded [-r, r], r=1.0 by default.
            generator = ImbalancedMixGaussGenerator(k=self.k, d=self.d, n=self.n, r=self.r)
            data = generator.load_data()
            data = self.split(data, config['T'])
        elif config["dataset"] == 'taxi':
            self.n = config["n"]
            generator = TaxiLoader(self.n, config['T'])
            # overwrite config and return data
            data = generator.load_data()
        elif config["dataset"] == 'loan':
            self.n = config["n"]
            generator = LoanLoader(self.n, config['T'])
            # overwrite config and return data
            data = generator.load_data()
            config['n'] = generator.n
            config['d'] = generator.d
        elif config["dataset"] == 'letter':
            self.n = config["n"]
            generator = LetterLoader(self.n, config['T'])
            # overwrite config and return data
            data = generator.load_data()
            config['n'] = generator.n
            config['d'] = generator.d
        else:
            raise NotImplementedError
        return data

    def split(self, data: np.arange, T: int):
        '''
        :param data: generated data
        :param T: number of parties in vertical setting
        :return: a list of dataset splited vertically
        '''
        num_attr = int(self.d / T)
        splited_data = []
        i = 0
        for i in range(T):
            splited_data.append(data[:, i * num_attr:(i + 1) * num_attr])
        np.append(splited_data[-1], data[:, (i + 1) * num_attr:])
        self.logger.info("Splited data shapes: {}".format([tmp.shape for tmp in splited_data]))
        return splited_data