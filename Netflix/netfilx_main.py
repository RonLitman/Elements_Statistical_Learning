import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# CONST

def load_and_set_data():
    '''
    loads and set the data
    :return:
    '''
    movie_titles = pd.read_csv('movie_titles.txt', sep=",", header=None)

    train_y_rating = pd.read_csv('train_y_rating.txt', delimiter=r"\s+", header=None)
    train_y_date = pd.read_csv('train_y_date.txt', delimiter=r"\s+", header=None)
    train_ratings_all = pd.read_csv('train_ratings_all.txt', delimiter=r"\s+", header=None)
    train_dates_all = pd.read_csv('train_dates_all.txt', delimiter=r"\s+", header=None)

    test_y_date = pd.read_csv('test_y_date.txt', delimiter=r"\s+", header=None)
    test_ratings_all = pd.read_csv('test_ratings_all.txt', delimiter=r"\s+", header=None)
    test_dates_all = pd.read_csv('test_dates_all.txt', delimiter=r"\s+", header=None)

    train_ratings_all.columns = movie_titles.iloc[:, 1]
    test_ratings_all.columns = movie_titles.iloc[:, 1]

    return movie_titles, train_y_rating, train_y_date, train_ratings_all, train_dates_all, test_y_date,\
           test_ratings_all, test_dates_all

def print_general_info(train_ratings_all, test_ratings_all, train_y_date, test_y_date):
    '''
    print general info for the data sets
    :return:
    '''
    print ('Number of movies rated: {}'.format(train_ratings_all.shape[1]))
    print('Number of user in train: {}'.format(train_ratings_all.shape[0]))
    print('Number of user in test: {}'.format(test_ratings_all.shape[0]))
    print('Number of missing ratings in train is: {} out of {}'.format())
    print('Date range in train Ratings: {} - {}'.format(int(train_y_date.min()), int(train_y_date.max())))
    print('Date range in test Ratings: {} - {}'.format(int(test_y_date.min()), int(test_y_date.max())))


movie_titles, train_y_rating, train_y_date, train_ratings_all, train_dates_all,\
test_y_date, test_ratings_all, test_dates_all = load_and_set_data()

print_general_info(train_ratings_all, test_ratings_all, train_y_date, test_y_date)