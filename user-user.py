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

matrix = coo_matrix((train_df['rating'], (train_df['book_coo'], train_df['user_coo'])), dtype=np.float32)
user_mean = np.array((matrix.sum(axis=0) / matrix.getnnz(axis=0)).tolist()[0])
train_df['user_mean'] = train_df.apply(lambda row: user_mean[row['user_coo']], axis=1)
expand_user_mean = coo_matrix((train_df['user_mean'], (train_df['book_coo'], train_df['user_coo'])), dtype=np.float32)
norm_matrix = matrix - expand_user_mean
cosine_matrix = cosine_similarity(norm_matrix.T, norm_matrix.T)
np.fill_diagonal(cosine_matrix, 1)

for k in [1, 5, 10, 50, 100]:
    os.makedirs('./result/user-user-{}'.format(k), exist_ok=True)
    for uid in tqdm(user_ids[:], desc='k={}'.format(k)):
        user_coo = uidd[uid]
        cosine = cosine_matrix[user_coo]
        indices = np.where(cosine != 0)[0]
        uid_test_df = test_df[test_df['user_id'] == uid]
        uid_mean = user_mean[user_coo]
        predict = list()
        for bid in uid_test_df['book_id'].values:
            bid_train_df = train_df[train_df['book_id'] == bid]
            intersection = list(set(bid_train_df['user_coo'].to_list()).intersection(set(indices.tolist())))
            if len(intersection) > 0:
                intersection = np.array(intersection)
                choices = np.argsort(cosine[intersection])[::-1][:k]
                used_cosine = cosine[intersection][choices]
                used_rating = bid_train_df[bid_train_df['user_coo'].isin(intersection[choices])]['rating'].values
                predict.append(used_cosine.dot(used_rating) / (np.abs(used_cosine).sum() + 1e-8) + uid_mean)
            else:
                predict.append(uid_mean)
        uid_test_df['predict'] = np.array(predict)
        uid_test_df.to_csv('./result/user-user-{}/{}.csv'.format(k, uid), index=False)