import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import skew
from scipy.special import boxcox1p
import copy
import numpy as np


class PreProcessor(object):
    """
    PreProcessData
    """

    def __init__(self):
        """
        Contructor
        """
        # File Path
        self.train_data_path = './Data/train/train.csv'
        self.test_data_path = './Data/test/test.csv'

        # Standardization
        self.standard_X = StandardScaler()

        # Tuning Parameters
        self.skewness_threshold = 0.75
        self.skewness_lam = 0.15

        self.train_data_split = 100  # 70%

        self.drop_columns = ['ID']


    def get_train_data(self):
        """
        前処理済の学習データを取得する。
        :return:
        """
        train_data = pd.read_csv(self.train_data_path)

        train_data = train_data.drop(columns=self.drop_columns)

        train_targets = train_data['Score'].values
        train_data = train_data.drop(columns=['Score'])

        # return self.standard_X.fit_transform(train_data), train_targets
        return train_data, train_targets

    def get_test_data(self):
        """
        前処理済のテストデータを取得する。
        :return:
        """
        test_data = pd.read_csv(self.test_data_path)
        test_data = test_data.drop(columns=self.drop_columns)
        return test_data

        
if __name__ == '__main__':
    pre_process = PreProcessor()



