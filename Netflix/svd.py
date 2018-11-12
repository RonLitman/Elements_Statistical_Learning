import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import SVD, evaluate
from surprise import NMF


def get_full_df():
    columns = ['uid', 'iid', 'rating', 'date']
    # ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
    ratings_df_train = pd.read_csv(
        '/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/df_train.csv')
    ratings_df_test = pd.read_csv(
        '/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/df_test.csv')
    df_train = ratings_df_train[columns]
    df_test = ratings_df_test[columns]
    for i in range(10000):
        temp = [i, 100, ratings_df_train[ratings_df_train.uid == i].y_rating.unique()[0],
                ratings_df_train[ratings_df_train.uid == i].y_date.unique()[0]]
        df_train = df_train.append(pd.DataFrame([temp], columns=columns))
    print('finished editing df train')
    df_test['uid'] = df_test['uid'] + 10000
    df = df_train.append(df_test)
    df.to_csv('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/df_join.csv')

def hide_y(df, size=0.2):
    pass


# df = get_full_df()

df = pd.read_csv('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/df_join.csv')

reader = Reader(rating_scale=(0.5,5.0))
data = Dataset.load_from_df(df[['uid', 'iid', 'rating']], reader)


# Split data into 5 folds

print('Split data into 5 folds')
data.split(n_folds=5)

# svd
print('SVD')
algo = SVD()

evaluate(algo, data, measures=['RMSE'])

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.train(trainset)
userid = str(10000)
itemid = str(100)

print(algo.predict(userid, 302))

# nmf
print('NMF')
algo = NMF()
evaluate(algo, data, measures=['RMSE'])

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.train(trainset)
userid = str(10000)
itemid = str(100)

print(algo.predict(userid, 302))