import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import models
from sklearn.metrics import mean_squared_error
import math
from sklearn.neighbors import KNeighborsClassifier
import more_itertools


def load_and_set_data():
    '''
    loads and set the data
    :return:
    '''
    train = {}
    test = {}

    movie_titles = pd.read_csv('movie_titles.txt', sep=",", header=None)

    train['train_y_rating'] = pd.read_csv('train_y_rating.txt', delimiter=r"\s+", header=None)
    train['train_y_date'] = pd.read_csv('train_y_date.txt', delimiter=r"\s+", header=None)
    train['train_ratings_all'] = pd.read_csv('train_ratings_all.txt', delimiter=r"\s+", header=None)
    train['train_dates_all'] = pd.read_csv('train_dates_all.txt', delimiter=r"\s+", header=None)

    test['test_y_date'] = pd.read_csv('test_y_date.txt', delimiter=r"\s+", header=None)
    test['test_ratings_all'] = pd.read_csv('test_ratings_all.txt', delimiter=r"\s+", header=None)
    test['test_dates_all'] = pd.read_csv('test_dates_all.txt', delimiter=r"\s+", header=None)

    train['train_ratings_all'].columns = movie_titles.iloc[:, ]
    test['test_ratings_all'].columns = movie_titles.iloc[:, ]

    return movie_titles, train, test


def set_df_from_metrix(train, test):
    print('Setting up DF from matrix')
    columns = ['uid', 'iid', 'rating', 'date', 'y_rating', 'y_date']
    df_train = pd.DataFrame(columns=columns)

    for i in range(train['train_ratings_all'].shape[0]):
        for j in range(train['train_ratings_all'].shape[1]):
            temp = []
            temp.append(i)
            temp.append(j)
            temp.append(train['train_ratings_all'].iloc[i, j])
            temp.append(train['train_dates_all'].iloc[i, j])
            if 0 in temp[2:]:
                continue
            temp.append(int(train['train_y_rating'].iloc[i]))
            temp.append(int(train['train_y_date'].iloc[i]))
            df_train = df_train.append(pd.DataFrame([temp], columns=columns))
        print('Finished user {} in training data'.format(i))

    df_train.to_csv(
        '/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/df_train.csv')

    columns = ['uid', 'iid', 'rating', 'date', 'y_date']
    df_test = pd.DataFrame(columns=columns)
    for i in range(test['test_ratings_all'].shape[0]):
        for j in range(test['test_ratings_all'].shape[1]):
            temp = []
            temp.append(i)
            temp.append(j)
            temp.append(test['test_ratings_all'].iloc[i, j])
            temp.append(test['test_dates_all'].iloc[i, j])
            if 0 in temp[2:]:
                continue
            # temp.append(int(test['test_y_rating'].iloc[i]))
            temp.append(int(test['test_y_date'].iloc[i]))
            df_test = df_test.append(pd.DataFrame([temp], columns=columns))
            print('Finished user {} in test data'.format(i))

    df_test.to_csv(
        '/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/df_test.csv')

    return df_train, df_test


def print_general_info(train, test):
    '''
    print general info for the data sets
    :return:
    '''

    n_user_train = train['train_ratings_all'].shape[0]
    n_user_test = test['test_ratings_all'].shape[0]
    n_movies = train['train_ratings_all'].shape[1]
    missing_ratings_train = (train['train_dates_all'].isin([0]).sum(axis=1))
    missing_ratings_test = (test['test_dates_all'].isin([0]).sum(axis=1))

    print('Number of movies rated: {}'.format(n_movies))
    print('Number of user in train: {}'.format(n_user_train))
    print('Number of user in test: {}'.format(n_user_test))

    print('Number of missing ratings in train is: {} out of {}'.format(sum(missing_ratings_train),
                                                                       n_user_train * n_movies))
    print('Mean Number of missing ratings (per user) in train is: {} %'.format(np.mean((missing_ratings_train))))
    print(
        'Number of missing ratings in test is: {} out of {}'.format(sum(missing_ratings_test), n_user_test * n_movies))
    print('Mean Number of missing ratings (per user) in train is: {} %'.format(np.mean(missing_ratings_test)))

    print('Date range in train Ratings: {} - {}'.format(int(train['train_y_date'].min()),
                                                        int(train['train_y_date'].max())))
    print('Date range in test Ratings: {} - {}'.format(int(test['test_y_date'].min()), int(test['test_y_date'].max())))


def clean_data(train, test):
    """
    remove users with variance lower then the threshold (in ratings)
    :param test:
    :param train:
    :param threshold:
    :return:
    """

    avg_before_1990, avg_before_2000, avg_after_2000 = get_avg_by_year(train['train_ratings_all'])
    avgTest_before_1990, avgTest_before_2000, Testavg_after_2000 = get_avg_by_year(test['test_ratings_all'])

    train['train_ratings_all'] = fill_missing(train['train_ratings_all'])
    test['test_ratings_all'] = fill_missing(test['test_ratings_all'])

    train['train_ratings_all'] = set_dates_feat(train, kind='train')
    test['test_ratings_all'] = set_dates_feat(test, kind='test')

    train['train_ratings_all']['avg_before_1990'] = avg_before_1990
    train['train_ratings_all']['avg_before_2000'] = avg_before_2000
    train['train_ratings_all']['avg_after_2000'] = avg_after_2000
    test['test_ratings_all']['avg_before_1990'] = avgTest_before_1990
    test['test_ratings_all']['avg_before_2000'] = avgTest_before_2000
    test['test_ratings_all']['avg_after_2000'] = Testavg_after_2000

    return train, test


def split_train_dev(data, labels, test, test_size=0.2, Type='random'):
    if Type == 'knn':
        data = data.replace(np.nan, 0)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(data, labels)
        index_for_dev = knn.kneighbors(test['test_ratings_all'].replace(np.nan, 0))[1]
        index_for_dev = [int(i[0]) for i in index_for_dev]
        index_for_train = [i for i in data.index if i not in index_for_dev]
        x_train = data.iloc[index_for_train, :]
        x_dev = data.iloc[index_for_dev, :]
        y_train = labels.iloc[index_for_train, :]
        y_dev = labels.iloc[index_for_dev, :]
    if Type == 'random':
        x_train, x_dev, y_train, y_dev = train_test_split(data, labels, test_size=test_size)
    return x_train, x_dev, y_train, y_dev


def scale_min_max(x):
    return (x - x.min()) / (x.max() - x.min())


def lin_model(x_train, x_test, y_train):
    print('\n')
    clf = models.lin_model(x_train, y_train)
    preds = clf.predict(x_test)

    return list(more_itertools.flatten(preds))


def set_column_order(df_ratings_all):
    coulmns_order = df_ratings_all.astype(bool).sum(axis=0).sort_values(ascending=False).index.tolist()
    df_ratings_all = df_ratings_all[coulmns_order]
    return df_ratings_all


def fill_missing(df_ratings_all):
    set_column_order(df_ratings_all)
    for i in range(14, df_ratings_all.shape[1]):
        column_name = df_ratings_all.columns[i]
        train_temp = df_ratings_all[df_ratings_all[column_name] > 0].iloc[:, list(range(0, i))]
        test_temp = df_ratings_all[df_ratings_all[column_name] == 0].iloc[:, list(range(0, i))]
        y_temp = df_ratings_all[df_ratings_all[column_name] > 0].iloc[:, [i]]

        fill_na = lin_model(train_temp, test_temp, y_temp)

        df_ratings_all[column_name][df_ratings_all[column_name] == 0] = fill_na
        df_ratings_all[column_name][df_ratings_all[column_name] > 5] = 5
        df_ratings_all[column_name][df_ratings_all[column_name] < 1] = 1

    return df_ratings_all


def set_dates_feat(dict, kind='train'):
    df_ratings_all = dict['{}_ratings_all'.format(kind)].copy()
    df_y_date = dict['{}_y_date'.format(kind)].copy()
    df_dates_all = dict['{}_dates_all'.format(kind)].copy()

    for i in range(df_dates_all.shape[1]):
        df_y_date[i] = df_y_date[0]

    df_dates_all = df_dates_all.replace(0, np.nan)

    seen_on_same_day = ((df_dates_all - df_y_date) == 0)

    rating_on_same_day = pd.DataFrame(np.array(seen_on_same_day) * np.array(df_ratings_all))
    rating_on_same_day = rating_on_same_day.replace(0, np.nan)
    df_ratings_all['day average'] = rating_on_same_day.mean(axis=1)

    df_ratings_all['movies_on_day'] = seen_on_same_day.sum(axis=1)
    df_ratings_all['date'] = df_y_date[0]
    return df_ratings_all


def get_avg_by_year(df_ratings_all):
    df_ratings_all = df_ratings_all.replace(0, np.NaN)
    avg_before_1990 = df_ratings_all[([i for i in df_ratings_all.columns[:99] if int(i[0]) <= 1990])].mean(
        axis=1).mean()
    avg_before_2000 = df_ratings_all[([i for i in df_ratings_all.columns[:99] if 2000 >= int(i[0]) > 1990])].mean(
        axis=1).mean()
    avg_after_2000 = df_ratings_all[([i for i in df_ratings_all.columns[:99] if int(i[0]) > 2000])].mean(axis=1).mean()
    df_ratings_all = df_ratings_all.replace(np.NaN, 0)
    return avg_before_1990, avg_before_2000, avg_after_2000
