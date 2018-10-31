import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from catboost import CatBoostRegressor
import lightgbm as lgb

def lin_model(x_train, y_train):
    '''
    fit a LinearRegression
    :param x_train:
    :param y_train:
    :return:
    '''
    print('Training a Linear Regression model')
    clf_reg = linear_model.LinearRegression()
    clf_reg.fit(x_train, y_train)
    return clf_reg

def run_catbost(x_train, x_dev, y_train, y_dev):
    '''

    :param x_train:
    :param x_dev:
    :param y_train:
    :param y_dev:
    :return:
    '''
    print('Training a CatBoost Regressor model')
    cb_model = CatBoostRegressor(iterations=500,
                                 learning_rate=0.05,
                                 depth=10,
                                 eval_metric='RMSE',
                                 random_seed=42,
                                 bagging_temperature=0.2,
                                 od_type='Iter',
                                 metric_period=50,
                                 od_wait=20)

    cb_model.fit(x_train, y_train,
                 eval_set=(x_dev, y_dev),
                 use_best_model=True,
                 verbose=True)
    # pred_test_cat = np.expm1(cb_model.predict(df_test))

    return cb_model

def run_lgb(x_train, x_dev, y_train, y_dev):
    '''

    :param x_train:
    :param x_dev:
    :param y_train:
    :param y_dev:
    :return:
    '''
    print('Training a lightgbm model')
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 40,
        "learning_rate": 0.005,
        "bagging_fraction": 0.6,
        "feature_fraction": 0.6,
        "bagging_frequency": 6,
        "bagging_seed": 42,
        "verbosity": -1,
        "seed": 42
    }

    lgtrain = lgb.Dataset(x_train, label=y_train.values.ravel())
    lgval = lgb.Dataset(x_dev, label=y_dev.values.ravel())
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000,
                      valid_sets=[lgtrain, lgval],
                      early_stopping_rounds=100,
                      verbose_eval=150,
                      evals_result=evals_result)

    # pred_test_y = np.expm1(model.predict(df_test, num_iteration=model.best_iteration))
    return model