import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


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

    train['train_ratings_all'].columns = movie_titles.iloc[:, 1]
    test['test_ratings_all'].columns = movie_titles.iloc[:, 1]

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


def clean_train(train, threshold=0, minimum_values=50, test_size=0.3):
    """
    remove users with missing data
    """

    x_train, x_dev, y_train, y_dev = split_train_dev(train['train_ratings_all'], train['train_y_rating'], test_size=test_size)
    missing_data = [x_train.astype(bool).sum(axis=1) > minimum_values][0]
    x_train = x_train[missing_data]
    y_train = y_train[missing_data]


    #  Replacing the 0  with NaN
    x_train = x_train.replace(0, np.nan)
    x_dev = x_dev.replace(0, np.nan)

    #  Replacing the missing values with the mean rating of the user
    x_train = x_train.apply(lambda x: x.replace(np.nan, x.mean()), axis=0)
    x_dev = x_dev.apply(lambda x: x.replace(np.nan, x.mean()), axis=0)


    # Adding std as feature
    x_train['std'] = x_train.std(axis=1)
    x_dev['std'] = x_dev.std(axis=1)
    return x_train, x_dev, y_train, y_dev


def split_train_dev(data, labels, test_size=0.4):
    x_train, x_dev, y_train, y_dev = train_test_split(data, labels, test_size=test_size)
    return x_train, x_dev, y_train, y_dev


def scale_min_max(x):
    return (x - x.min()) / (x.max() - x.min())


def clean_test(test, threshold=0):
    """
    replace 0 with user's mean rate
    """

    test['test_ratings_all'] = test['test_ratings_all'].replace(0, np.nan)

    # Replacing the missing values with the mean rating of the user
    test['test_ratings_all'] = test['test_ratings_all'].apply(lambda x: x.replace(np.nan, x.mean()), axis=0)

    # Adding std as feature
    test['test_ratings_all']['std'] = test['test_ratings_all'].std(axis=1)

    return test
