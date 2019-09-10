from preprocess import PreProcessor
from train import Trainer
from predict import Predictor
import sys


class Runner(object):
    """
    全体の処理を実行させる
    """
    __this_file_name = sys.modules[__name__].__name__

    def __init__(self):
        """
        constructor
        """
        self.__pre_processor = PreProcessor()
        self.__trainer = Trainer()
        self.__predictor = Predictor()

    def run(self):
        """
        処理を実行する
        :return:
        """
        # create model
        # model = self.__trainer.fit_model()
        model = self.__trainer.adaboost()

        # predict
        predict_data = self.__predictor.predict(model)
        print(predict_data)

        self.__predictor.write_file_submit(predict_data)


if __name__ == '__main__':
    runner = Runner()
    runner.run()
