from preprocess import PreProcessor
import pandas as pd
import datetime
import numpy as np


class Predictor(object):
    """
    Predict処理を実行する
    """

    def __init__(self):
        """
        constructor
        """
        # preprocessor instance
        self.__pre_process = PreProcessor()

    def predict(self, model):
        """
        Predict and write data for submit
        :param model:
        :return:
        """
        test = self.__pre_process.get_test_data()

        predict_data = model.predict(test)

        return predict_data

    @staticmethod
    def write_file_submit(predict_data):
        """
        Write predict data to submit file
        :param predict_data:
        :return:
        """
        submit = pd.read_csv("./Data/sample_submit.csv", header=None)
        submit[1] = predict_data
        print(submit[1])
        now = datetime.datetime.now()
        now_str = '{}_{}_{}_{}_{}'.format(
            now.year, now.month, now.day, now.hour, now.minute)
        submit_file = './Data/submit/submit_{}.csv'.format(now_str)
        submit.to_csv(submit_file, header=None, index=None)

    @staticmethod
    def essemble_results(file1, file2):

        data1 = pd.read_csv(file1, sep="\t", header=None)
        data2 = pd.read_csv(file2, sep="\t", header=None)
        predict_data = (np.array(data1[1]) + np.array(data2[1])) / 2

        return predict_data


if __name__ == '__main__':
    predictor = Predictor()
    predict_data = predictor.essemble_results('./Data/submit/submit_2019_8_19_23_0.csv',
                                              './Data/submit/submit_2019_8_20_23_11.csv')

    predictor.write_file_submit(predict_data)
