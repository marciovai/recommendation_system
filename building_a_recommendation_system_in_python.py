# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import zipfile
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# Download MovieLens data.
print("Downloading movielens data...")

urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
zip_ref = zipfile.ZipFile('movielens.zip', "r")
zip_ref.extractall()
print("Done. Dataset contains:")
print(zip_ref.read('ml-100k/u.info'))

# Load each data set (users, movies, and ratings).
ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

# The movies file contains a binary feature for each genre.
genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movies_cols = [
    'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
] + genre_cols

movies = pd.read_csv(
    'ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

# Since the ids start at 1, we shift them to start at 0.
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x-1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x-1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x-1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))
ratings['date'] = pd.to_datetime(ratings['unix_timestamp'], origin='unix', unit='s').dt.date

ratings.drop('unix_timestamp', axis=1, inplace=True)

data_pivot = ratings[['user_id', 'movie_id', 'rating']].pivot(index='user_id', columns='movie_id').fillna(0).astype(int)  

# create scipy sparse from pivot table
data_sparse = sparse.csr_matrix(data_pivot)

similarities_sparse = cosine_similarity(data_sparse, dense_output=False)

# returns index (column position) of top n similarities in each row
def top_n_idx_sparse(matrix, n):
    '''Return index of top n values in each row of a sparse matrix'''
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx
user_user_similar = top_n_idx_sparse(similarities_sparse, 5)

# transforms result from sparse matrix into a dict user: [job1, job2]
user_user_similar_dict = {}
for idx, val in enumerate(user_user_similar):
      user_user_similar_dict.update({idx: val.tolist()})

# gets actual user ids from data based on sparse matrix position index
similar_users_final = {}
for user, similar_users in user_user_similar_dict.items():
    idx = data_pivot.index[user]
    values = []
    for value in similar_users:
        values.append(data_pivot.index[value])

    similar_users_final.update({idx: values})


# transforms list of users: [similar_user1, similar_user2] into list of user: [job1, job2]
user_movies = {}
for user, similar_users in similar_users_final.items():
    # remove the user itself from similar_users (since cos_sim(user_1, user_1) is 1)
    try:
        del similar_users[similar_users.index(user)]
    except:
        pass
    # get movie ids from list of movies rated by similar users.
    # also apply extra logic to get the most high rated movies from the similar users
    movies_rec = ratings[(ratings['user_id'].isin(similar_users)) & ratings['rating']>=3]
    if movies_rec.empty:
      movies_rec = ratings[(ratings['user_id'].isin(similar_users)) & ratings['rating']>=2]
    if movies_rec.empty:
      movies_rec = ratings[(ratings['user_id'].isin(similar_users)) & ratings['rating']>=1]
    movies_sample = movies_rec.sample(n=10, random_state=33)['movie_id'].values
    user_movies.update({user: list(set(movies_sample))})

# transform dictionary into list of tuples and save on DataFrame
user_movie_tuple = [(user, movie) for user, user_movies in user_movies.items() for movie in user_movies]
user_movie_df = pd.DataFrame(user_movie_tuple, columns=['user_id', 'movie_id'])

rec_batch = pd.merge(user_movie_df, movies[['movie_id', 'title']], on='movie_id', how='left')

rec_batch[rec_batch['user_id']=='99']

