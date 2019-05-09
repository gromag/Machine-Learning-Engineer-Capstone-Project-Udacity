import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from random import sample, seed


class DataSampler():
    """Class responsible for holding, sampling and splitting the training in train and validation set"""

    def __init__(self, data=None, sample_percent = 1):
        """

        """
        super().__init__()

        assert (data is not None), "No data passed"

        try:
            self.RANDOM_SEED = RANDOM_SEED
        except:
            self.RANDOM_SEED = 1234

        assert(sample_percent > 0 and sample_percent <=1)

        self._data = data

        if sample_percent < 1:
            sample_size = math.ceil(len(data) * sample_percent)
            self._data = data.sample(sample_size)

    def data(self):
        return self._data


    def train_test_split(self, X, y, test_size = 0.2):
        """
        Split train data into `train` `test` set

        Input:
        ------
        X:
        y:
        test_size:

        Returns:
        ------
        (X_train, X_test, y_train, y_test)

        """
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=1)

        return (X_train, X_valid, y_train, y_valid)

    def train_test_split_by_columns(self, X_columns, y_columns, rows_filter = None, sample_percent = 1, test_size = 0.2):

        """
        Split train data into `train` `test` set

        Input:
        ------
        X_columns:
        y_column:
        rows_filter:
        test_size:
        sample_percent: float 0.0 < n <= 1.0 percentage of samples


        Returns:
        ------
        (X_train, X_test, y_train, y_test)

        """
        X = self.data[X_columns]
        y = self.data[y_columns]
        
        assert(len(X) == len(y))
        assert(sample_percent > 0 and sample_percent <=1)


        assert(rows_filter is None or len(rows_filter) == len(X))

        if rows_filter is not None:
            X = X[rows_filter]
            y = y[rows_filter]

        if sample_percent < 1:
            sample_size = math.ceil(len(X) * sample_percent)
            X = X.sample(sample_size, random_state=self.RANDOM_SEED)
            y = y.loc[X.index]

        return self.train_test_split(X, y, test_size)