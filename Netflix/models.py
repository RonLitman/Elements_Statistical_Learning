import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from catboost import CatBoostRegressor
import lightgbm as lgb
import math
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
import xgboost as xgb


def lin_model(x_train, y_train, method='normal'):
    '''
    fit a LinearRegression
    :param x_train:
    :param y_train:
    :return:
    '''
    print('Training a Linear Regression model')

    if method == 'normal':
        clf_reg = linear_model.LinearRegression()
    if method == 'Lasso':
        clf_reg = linear_model.Lasso(alpha=0.1)

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
                      early_stopping_rounds=200,
                      verbose_eval=150,
                      evals_result=evals_result)

    # pred_test_y = np.expm1(model.predict(df_test, num_iteration=model.best_iteration))
    return model

def run_xgb(x_train, x_dev, y_train, y_dev):
    print('Training a xgb model')
    params = {'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'eta': 0.001,
              'max_depth': 10,
              'subsample': 0.6,
              'colsample_bytree': 0.6,
              'alpha': 0.001,
              'random_state': 42,
              'silent': True}

    tr_data = xgb.DMatrix(x_train, y_train)
    va_data = xgb.DMatrix(x_dev, y_dev)

    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds=100, verbose_eval=1)

    return model_xgb

def run_knn_cosine(x_train, x_dev, y_train, y_dev, k=50):
    '''

    :param x_train:
    :param x_dev:
    :param y_train:
    :param y_dev:
    :param k:
    :return:
    '''
    print('\n')
    print('Training a KNN model with cosine_similarity for k = {}'.format(k))
    x_train_temp = x_train.reset_index(drop=True)
    x_dev_temp = x_dev.reset_index(drop=True)
    y_train_temp = y_train.reset_index(drop=True)
    y_dev_temp = y_dev.reset_index(drop=True)

    preds_dev = np.zeros((y_dev_temp.shape[0],1))
    # similarity_dist = 1 - cosine_similarity(x_train_temp, x_dev_temp)
    similarity_dist = cosine_similarity(x_train_temp, x_dev_temp)
    for i in range(similarity_dist.shape[1]):
        # min_ind_dev = similarity_dist[:,1].argsort()[:k]
        # preds_dev[i] = np.mean(y_train_temp[y_train_temp.index.isin(min_ind_dev)])
        preds_dev[i] = np.sum((y_train_temp * similarity_dist[:,1].reshape(similarity_dist[:,1].shape[0],1))) /\
                       np.sum(similarity_dist[:,1])

    mse_dev_reg = math.sqrt(mean_squared_error(y_dev_temp, preds_dev))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))

def run_knn_manhattan(x_train, x_dev, y_train, y_dev, k=50):
    '''

    :param x_train:
    :param x_dev:
    :param y_train:
    :param y_dev:
    :param k:
    :return:
    '''
    print('\n')
    print('Training a KNN model with manhattan for k = {}'.format(k))
    dist = DistanceMetric.get_metric('manhattan')
    x_train_temp = x_train.reset_index(drop=True)
    x_dev_temp = x_dev.reset_index(drop=True)
    y_train_temp = y_train.reset_index(drop=True)
    y_dev_temp = y_dev.reset_index(drop=True)

    preds_dev = np.zeros((y_dev_temp.shape[0],1))
    similarity_dist = dist.pairwise(x_train_temp, x_dev_temp)
    for i in range(similarity_dist.shape[1]):
        min_ind_dev = similarity_dist[:,1].argsort()[:k]
        preds_dev[i] = np.mean(y_train_temp[y_train_temp.index.isin(min_ind_dev)])

    mse_dev_reg = math.sqrt(mean_squared_error(y_dev_temp, preds_dev))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))


