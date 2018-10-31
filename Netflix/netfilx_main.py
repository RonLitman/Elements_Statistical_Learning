import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Netflix import models
from Netflix.general_function import *
from sklearn.metrics import mean_squared_error
# CONST

movie_titles, train, test = load_and_set_data()
print_general_info(train, test)
train, test = clean_data(train, test)
x_train, x_dev, y_train, y_dev = split_train_dev(train)

clf = models.lin_model(x_train, y_train)


preds_train = clf.predict(x_train)
preds_dev = clf.predict(x_dev)

mse_train_reg = mean_squared_error(y_train, preds_train)
mse_dev_reg = mean_squared_error(y_dev, preds_dev)

print('MSE on train is: {}'.format(mse_train_reg))
print('MSE on DEV is: {}'.format(mse_dev_reg))


clf = models.run_catbost(x_train, x_dev, y_train, y_dev)

preds_train = clf.predict(x_train)
preds_dev = clf.predict(x_dev)

mse_train_reg = mean_squared_error(y_train, preds_train)
mse_dev_reg = mean_squared_error(y_dev, preds_dev)

print('MSE on train is: {}'.format(mse_train_reg))
print('MSE on DEV is: {}'.format(mse_dev_reg))


clf = models.run_lgb(x_train, x_dev, y_train, y_dev)

preds_train = clf.predict(x_train, num_iteration=clf.best_iteration)
preds_dev = clf.predict(x_dev, num_iteration=clf.best_iteration)

mse_train_reg = mean_squared_error(y_train, preds_train)
mse_dev_reg = mean_squared_error(y_dev, preds_dev)

print('MSE on train is: {}'.format(mse_train_reg))
print('MSE on DEV is: {}'.format(mse_dev_reg))


preds_test = clf.predict(test['test_ratings_all'], num_iteration=clf.best_iteration)
np.savetxt('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/preds.csv',
           preds_test, delimiter=",")