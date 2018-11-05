import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import models
from general_function import *
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
    return clf, mse_dev_reg


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


def run_lgb_model(x_train, x_dev, x_unseen, y_train, y_dev, y_unseen):
    print('\n')
    clf = models.run_lgb(x_train, x_dev, y_train, y_dev)
    preds_train = clf.predict(x_train, num_iteration=clf.best_iteration)
    preds_dev = clf.predict(x_dev, num_iteration=clf.best_iteration)
    preds_unseen = clf.predict(x_unseen, num_iteration=clf.best_iteration)
    mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
    mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))
    mse_unseen_reg = math.sqrt(mean_squared_error(y_unseen, preds_unseen))
    print('RMSE on train is: {}'.format(mse_train_reg))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))
    print('RMSE on UNSEEN is: {}'.format(mse_unseen_reg))
    return clf


def run_xgb_model(x_train, x_dev, y_train, y_dev):
    print('\n')
    clf = models.run_xgb(x_train, x_dev, y_train, y_dev)
    dtest = xgb.DMatrix(x_dev)
    preds_dev = clf.predict(dtest, ntree_limit=clf.best_ntree_limit)
    mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))
    print('RMSE on DEV is: {}'.format(mse_dev_reg))
    return clf

# CONST

movie_titles, train, test = load_and_set_data()
print_general_info(train, test)
train, test = clean_data(train, test)
x_train, x_dev, y_train, y_dev = split_train_dev(train)

clf = models.lin_model(x_train, y_train)

preds_train = clf.predict(x_train)
preds_dev = clf.predict(x_dev)

mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))

print('RMSE on train is: {}'.format(mse_train_reg))
print('RMSE on DEV is: {}'.format(mse_dev_reg))

# clf = models.run_catbost(x_train, x_dev, y_train, y_dev)
#
# preds_train = clf.predict(x_train)
# preds_dev = clf.predict(x_dev)
#
# mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
# mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))
#
# print('RMSE on train is: {}'.format(mse_train_reg))
# print('RMSE on DEV is: {}'.format(mse_dev_reg))


clf = models.run_lgb(x_train, x_dev, y_train, y_dev)
test = clean_test(test)

mse_dev_list = []
for i in range(5):
    x_train, x_dev, y_train, y_dev = clean_train(train, minimum_values=80, test_size=0.3)
    clf, mse_dev_reg = run_lin_model(x_train, x_dev, y_train, y_dev)
    mse_dev_list.append(mse_dev_reg)
print('The linear regression ran {} times, the avarage RMSE is {}, Max RMSE is {}'.format(i, np.mean(mse_dev_list),
                                                                                          max(mse_dev_list)))

x_dev, x_unseen, y_dev, y_unseen = split_train_dev(x_dev, y_dev, test_size=0.5)
# clf = run_cat(x_train, x_dev, y_train, y_dev)

preds_train = clf.predict(x_train, num_iteration=clf.best_iteration)
preds_dev = clf.predict(x_dev, num_iteration=clf.best_iteration)

mse_train_reg = math.sqrt(mean_squared_error(y_train, preds_train))
mse_dev_reg = math.sqrt(mean_squared_error(y_dev, preds_dev))

print('RMSE on train is: {}'.format(mse_train_reg))
print('RMSE on DEV is: {}'.format(mse_dev_reg))
clf = run_lgb_model(x_train, x_dev, x_unseen, y_train, y_dev, y_unseen)

# train LGB with all the data before prediction
# x_train, x_dev, y_train, y_dev = split_train_dev(train['train_ratings_all'], train['train_y_rating'], test_size=0.3)
# clf = run_lgb_model(x_train, x_dev, x_unseen, y_train, y_dev, y_unseen)

preds_test = clf.predict(test['test_ratings_all'], num_iteration=clf.best_iteration)
# np.savetxt('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/preds.csv',
#            preds_test, delimiter=",")
