import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
from matplotlib import pyplot as plt
import json
from collections import Counter
import math
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings
tqdm.pandas()
warnings.filterwarnings('ignore')

train_df = pd.read_csv('./data/train_ratings.csv')
test_df = pd.read_csv('./data/test_ratings.csv')
user_ids = np.load('./data/user_ids.npy')
book_ids = np.load('./data/book_ids.npy')

uidd = dict()
for i, uid in enumerate(user_ids):
    uidd[uid] = i
bidd = dict()
for i, bid in enumerate(book_ids):
    bidd[bid] = i

train_df['user_coo'] = train_df.apply(lambda row: uidd[row['user_id']], axis=1)
train_df['book_coo'] = train_df.apply(lambda row: bidd[row['book_id']], axis=1)
matrix = coo_matrix((train_df['rating'], (train_df['user_coo'], train_df['book_coo'])), dtype=np.float32)
book_mean = np.array((matrix.sum(axis=0) / matrix.getnnz(axis=0)).tolist()[0])
train_df['book_mean'] = train_df.apply(lambda row: book_mean[int(row['book_coo'])], axis=1)
expand_book_mean = coo_matrix((train_df['book_mean'], (train_df['user_coo'], train_df['book_coo'])), dtype=np.float32)
norm_matrix = matrix - expand_book_mean
cosine_matrix = cosine_similarity(norm_matrix.T, norm_matrix.T)
np.fill_diagonal(cosine_matrix, 1)
for k in [1, 2, 3, 4, 5]:
    os.makedirs('./result/item-item-{}'.format(k), exist_ok=True)
    for uid in tqdm(user_ids[:], desc='k={}'.format(k)):
        user_coo = uidd[uid]
        uid_test_df = test_df[test_df['user_id'] == uid]
        uid_train_df = train_df[train_df['user_id'] == uid]
        predict = list()
        rated_bids = uid_train_df['book_id'].values
        rated_book_coo = np.array([bidd[_bid] for _bid in rated_bids])
        for bid in uid_test_df['book_id'].values:
            book_coo = bidd[bid]
            bid_mean = book_mean[book_coo]
            indices = np.argsort(cosine_matrix[book_coo, rated_book_coo])[::-1][:k]
            choices = rated_book_coo[indices]
            used_cosine = cosine_matrix[book_coo, choices]
            used_rating = uid_train_df[uid_train_df['book_coo'].isin(choices)]['rating'].values
            predict.append(used_cosine.dot(used_rating) / (np.abs(used_cosine).sum() + 1e-8) + bid_mean)
        uid_test_df['predict'] = np.array(predict)
        uid_test_df.to_csv('./result/item-item-{}/{}.csv'.format(k, uid), index=False)