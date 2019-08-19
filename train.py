# standard library
from datetime import date
import pandas as pd
import numpy as np

# scikit learn
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV

import xgboost as xgb
# import lightgbm as lgb


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
        self.__train, self.__y_train = self.__pre_process.get_train_data()

        # features = train_data.drop(columns=['keiyaku_pr'])
        # prices = train_data['keiyaku_pr'].values
        #
        # X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

        # Tuning Parameters
        self.__n_folds = 3  # Cross-validation with k-folds

        # Models
        self.__lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
        self.__ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
        self.__KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        self.__GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                                  max_depth=4, max_features='sqrt',
                                                  min_samples_leaf=15, min_samples_split=10,
                                                  loss='huber', random_state=5)
        self.__model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0,
                                            learning_rate=0.05, max_depth=6,
                                            min_child_weight=1.5, n_estimators=7200,
                                            reg_alpha=0.9, reg_lambda=0.6,
                                            subsample=0.2, seed=42, silent=1,
                                            random_state=7)
        # self.__model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
        #                                      learning_rate=0.05, n_estimators=720,
        #                                      max_bin=55, bagging_fraction=0.8,
        #                                      bagging_freq=5, feature_fraction=0.2319,
        #                                      feature_fraction_seed=9, bagging_seed=9,
        #                                      min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    def get_scores(self):
        """
        学習関数
        :return:
        """
        score = self.rmsle_cv(self.__lasso)
        print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = self.rmsle_cv(self.__ENet)
        print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = self.rmsle_cv(self.__KRR)
        print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = self.rmsle_cv(self.__GBoost)
        print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        score = self.rmsle_cv(self.__model_xgb)
        print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
        # score = self.rmsle_cv(self.__model_lgb)
        # print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


    def mean_absolute_percentage_error(self,y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def fit_model(self):
        """
        モデルをフィットする
        :return:
        """
        # model = self.train_model(self.__train, self.__y_train)
        test_size = 1/self.__n_folds
        # Split the training data into an extra set of test
        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(self.__train, self.__y_train,test_size = test_size,random_state=0)
        print(np.shape(x_train_split), np.shape(x_test_split), np.shape(y_train_split), np.shape(y_test_split))
        lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
                                0.3, 0.6, 1],
                        max_iter=50000, cv=10)
        # lasso = RidgeCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
        #                         0.3, 0.6, 1], cv=10)

        # lasso = ElasticNetCV(cv=10, random_state=0)

   


        # lasso.fit(x_train_split, y_train_split)
        # y_predicted = lasso.predict(X=x_test_split)
        # mape = self.mean_absolute_percentage_error(y_test_split,y_predicted)
        # print(mape)
     

        # xgboostモデルの作成
        reg = xgb.XGBRegressor()

        # ハイパーパラメータ探索
        reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
        reg_cv.fit(x_train_split, y_train_split)
        print(reg_cv.best_params_, reg_cv.best_score_)
        # 改めて最適パラメータで学習
        reg = xgb.XGBRegressor(**reg_cv.best_params_)
        reg.fit(x_train_split, y_train_split)


        # 学習モデルの保存、読み込み
        # import pickle
        # pickle.dump(reg, open("model.pkl", "wb"))
        # reg = pickle.load(open("model.pkl", "rb"))

        # 学習モデルの評価
        pred_train = reg.predict(x_train_split)
        pred_test = reg.predict(x_test_split)
        # print(self.mean_absolute_percentage_error(y_train_split, pred_train))
        print(self.mean_absolute_percentage_error(y_test_split, pred_test))

        # import pandas as pd
        # import matplotlib.pyplot as plt
        # importances = pd.Series(reg.feature_importances_, index = boston.feature_names)
        # importances = importances.sort_values()
        # importances.plot(kind = "barh")
        # plt.title("imporance in the xgboost Model")
        # plt.show()
        return reg

    def train_model(self, X, y):
        """ Performs grid search over the 'max_depth' parameter for a
            decision tree regressor trained on the input data [X, y]. """

        # Create cross-validation sets from the training data
        cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

        # Create a decision tree regressor object
        regressor = DecisionTreeRegressor()

        # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
        params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        # Transform 'performance_metric' into a scoring function using 'make_scorer'
        scoring_fnc = make_scorer(self.r2_score)

        # Create the grid search cv object --> GridSearchCV()
        grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

        # Fit the grid search object to the data to compute the optimal model
        grid = grid.fit(X, y)

        # Return the optimal model after fitting the data
        return grid.best_estimator_

    def rmsle_cv(self, model):
        """
        calculate rmse for cross validation
        :return:
        """
        kf = KFold(self.__n_folds, shuffle=True, random_state=42).get_n_splits(self.__train.values)
        rmse = np.sqrt(-cross_val_score(model, self.__train.values, self.__y_train, scoring="neg_mean_squared_error",
                                        cv=kf))
        return rmse

    @staticmethod
    def r2_score(y_true, y_predict):
        """ Calculates and returns the performance score between
                true (y_true) and predicted (y_predict) values based on the metric chosen. """

        score = r2_score(y_true, y_predict)

        # Return the score
        return score


if __name__ == '__main__':
    trainer = Trainer()
    trainer.fit_model()
