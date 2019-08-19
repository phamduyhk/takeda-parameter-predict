from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense

import numpy as np
import matplotlib.pyplot as plt

from preprocess import PreProcessor


class Trainer(object):
    """
    Train Class
    """

    def __init__(self):
        """
        Constructor
        """
        # preprocessor instance
        self.__pre_process = PreProcessor()
        self.__train_data, self.__train_targets = self.__pre_process.get_train_data()
        # print(self.__train_data)

        # Tuning Parameters
        self.__n_folds = 5  # Cross-validation with k-folds
        self.__num_epochs = 400

    def build_model(self):
        """
        モデル構築
        :return:
        """
        # NN model
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', kernel_initializer='normal', input_shape=(self.__train_data.shape[1], )))
        model.add(layers.Dense(256, activation='relu', kernel_initializer='normal'))
        model.add(layers.Dense(256, activation='relu', kernel_initializer='normal'))
        model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
        model.compile(optimizer='adam', loss="mse", metrics=['mape'])
        model.summary()
        return model

    def fit_model(self):
        """
        モデルをFitする
        :return:
        """
        # Kerasモデル構築(コンパイル済)
        model = self.build_model()

        # モデルをサイレントモード(verbose=0)で適合
        model.fit(self.__train_data, self.__train_targets, epochs=self.__num_epochs, batch_size=16, verbose=0)

        return model

    def evaluate_cross(self):
        """
        交差評価
        :return:
        """
        all_scores = []
        num_val_samples = int(len(self.__train_data) / self.__n_folds)

        for i in range(self.__n_folds):
            print('processing fold #  {}'.format(i))

            # 検証データの準備
            val_data = self.__train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self.__train_targets[i * num_val_samples: (i + 1) * num_val_samples]

            # 訓練データの準備
            partial_train_data = np.concatenate(
                [self.__train_data[:i * num_val_samples], self.__train_data[(i + 1) * num_val_samples:]], axis=0)
            partial_targets_data = np.concatenate(
                [self.__train_targets[:i * num_val_samples], self.__train_targets[(i + 1) * num_val_samples:]], axis=0)

            # Kerasモデル構築(コンパイル済)
            model = self.build_model()

            # モデルをサイレントモード(verbose=0)で適合
            model.fit(partial_train_data, partial_targets_data, epochs=self.__num_epochs, batch_size=16, verbose=0)

            # モデルを検証データで評価
            val_mse, val_mape = model.evaluate(val_data, val_targets, verbose=0)
            all_scores.append(val_mape)

        print(all_scores)

        return np.mean(all_scores)

    def visualize_k_folds(self):
        """
        k分割交差検証のvisualization
        :return:
        """
        all_mape_histories = []
        num_val_samples = int(len(self.__train_data) / self.__n_folds)

        for i in range(self.__n_folds):
            print('processing fold #  {}'.format(i))

            # 検証データの準備
            val_data = self.__train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self.__train_targets[i * num_val_samples: (i + 1) * num_val_samples]

            # 訓練データの準備
            partial_train_data = np.concatenate(
                [self.__train_data[:i * num_val_samples], self.__train_data[(i + 1) * num_val_samples:]], axis=0)
            partial_targets_data = np.concatenate(
                [self.__train_targets[:i * num_val_samples], self.__train_targets[(i + 1) * num_val_samples:]], axis=0)

            # Kerasモデル構築(コンパイル済)
            model = self.build_model()

            # モデルをサイレントモード(verbose=0)で適合
            history = model.fit(partial_train_data, partial_targets_data,
                                validation_data=(val_data, val_targets),
                                epochs=self.__num_epochs, batch_size=16, verbose=0)

            # モデルを検証データで評価
            mape_history = history.history['val_mean_absolute_percentage_error']
            all_mape_histories.append(mape_history)

        print(all_mape_histories)

        average_mape_history = [
            np.mean([x[i] for x in all_mape_histories]) for i in range(self.__num_epochs)]

        plt.plot(range(1, len(average_mape_history) + 1), average_mape_history)
        plt.xlabel('Epochs')
        plt.ylabel('Validation MAPE')
        plt.show()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.visualize_k_folds()
