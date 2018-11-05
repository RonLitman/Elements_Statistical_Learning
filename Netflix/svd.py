import pandas as pd
import numpy as np
import surprise

columns = ['uid', 'iid', 'rating']

# ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
ratings_df_train = pd.read_csv('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/df_train.csv')
ratings_df_test = pd.read_csv('/Users/ronlitman/Ronlitman/University/Statistic/שנה א׳ - סמט׳ א׳/למידה סטטיסטית/Netflix/df_test.csv')

df = ratings_df_train[columns].append()

reader = surprise.Reader(rating_scale=(0.5,5.0))

