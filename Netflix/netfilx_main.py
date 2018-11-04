import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Netflix import models
from Netflix.general_function import *
from sklearn.metrics import mean_squared_error
import math
import xgboost as xgb


def run_lin_model(x_train, x_dev, y_train, y_dev):
    print('\n')
    clf = models.lin_model(x_train, y_train)
    preds_train = clf.predict(x_train)
    preds_dev = clf.predict(x_dev)
    mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
    mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))
    print('RMSE on train is: {}'.format(mse_train_reg))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))
    return clf

def run_cat(x_train, x_dev, y_train, y_dev):
    print('\n')
    clf = models.run_catbost(x_train, x_dev, y_train, y_dev)
    preds_train = clf.predict(x_train)
    preds_dev = clf.predict(x_dev)
    mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
    mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))
    print('RMSE on train is: {}'.format(mse_train_reg))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))
    return clf

def run_lgb_model(x_train, x_dev, y_train, y_dev):
    print('\n')
    clf = models.run_lgb(x_train, x_dev, y_train, y_dev)
    preds_train = clf.predict(x_train, num_iteration=clf.best_iteration)
    preds_dev = clf.predict(x_dev, num_iteration=clf.best_iteration)
    mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
    mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))
    print('RMSE on train is: {}'.format(mse_train_reg))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))
    return clf

def run_xgb_model(x_train, x_dev, y_train, y_dev):
    print('\n')
    clf = models.run_xgb(x_train, x_dev, y_train, y_dev)
    dtest = xgb.DMatrix(x_dev)
    preds_dev = clf.predict(dtest, ntree_limit=clf.best_ntree_limit)
    mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))
    return clf


movie_titles, train, test = load_and_set_data()

print_general_info(train, test)

df_train, df_test = set_df_from_metrix(train, test)


# train, test = clean_data(train, test)

# x_train, x_dev, y_train, y_dev = split_train_dev(train['train_ratings_all'], train['train_y_rating'])

# clf = run_lin_model(x_train, x_dev, y_train, y_dev)

# clf = run_cat(x_train, x_dev, y_train, y_dev)

# models.run_knn_cosine(x_train, x_dev, y_train, y_dev)
# models.run_knn_manhattan(x_train, x_dev, y_train, y_dev)

# clf = run_xgb_model(x_train, x_dev, y_train, y_dev)

# clf = run_lgb_model(x_train, x_dev, y_train, y_dev)


# preds_test = clf.predict(test['test_ratings_all'], num_iteration=clf.best_iteration)
# np.savetxt('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/preds.csv',
#            preds_test, delimiter=",")