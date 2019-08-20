#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xgboost as xgb

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from preprocess import PreProcessor

"""XGBoost で学習の履歴を可視化するサンプルコード"""


def main():
    pre_process = PreProcessor()
    X, y = pre_process.get_train_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        random_state=42,
                                                        stratify=y)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
    }

    # 学習時に用いる検証用データ
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    # 学習過程を記録するための辞書
    evals_result = {}
    bst = xgb.train(xgb_params,
                    dtrain,
                    num_boost_round=1000,  # ラウンド数を増やしておく
                    evals=evals,
                    evals_result=evals_result,
                    )

    y_pred_proba = bst.predict(dtest)
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)

    # 学習の課程を折れ線グラフとしてプロットする
    train_metric = evals_result['train']['logloss']
    plt.plot(train_metric, label='train logloss')
    eval_metric = evals_result['eval']['logloss']
    plt.plot(eval_metric, label='eval logloss')
    plt.grid()
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('logloss')
    plt.show()


if __name__ == '__main__':
    main()
