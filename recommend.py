import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

user_w = np.load('./learn/matrix-factorization-50/user_w.npy')
user_b = np.load('./learn/matrix-factorization-50/user_b.npy')
book_w = np.load('./learn/matrix-factorization-50/book_w.npy')
book_b = np.load('./learn/matrix-factorization-50/book_b.npy')
book_ids = np.load('./data/book_ids.npy')
user_ids = np.load('./data/user_ids.npy')
train_ratings = pd.read_csv('./data/train_ratings.csv')
test_ratings = pd.read_csv('./data/test_ratings.csv')
mu = train_ratings['rating'].mean()
os.makedirs('./recommend/matrix-factorization', exist_ok=True)
for user_coo in tqdm(range(user_ids.shape[0]), desc='matrix-factorization'):
    uid = user_ids[user_coo]
    ratings = user_w[user_coo] @ book_w.T + user_b[user_coo] * book_b + mu
    rated_bid = train_ratings[train_ratings['user_id'] == uid]['book_id'].values
    unrated_book_coo = np.where(~np.isin(book_ids, rated_bid))[0]
    unrated_predict = ratings[unrated_book_coo]
    recommend_book_coo = np.argsort(unrated_predict)[::-1][:100]
    np.save('./recommend/matrix-factorization/{}.npy'.format(uid), book_ids[unrated_book_coo][recommend_book_coo])

user_w = np.load('./learn/content-based-1.0/user_w.npy')
user_b = np.load('./learn/content-based-1.0/user_b.npy')
encode_books = np.load('./data/encode_books.npy')
os.makedirs('./recommend/content-based', exist_ok=True)
for user_coo in tqdm(range(user_ids.shape[0]), desc='content-based'):
    uid = user_ids[user_coo]
    ratings = user_w[user_coo] @ encode_books.T + user_b[user_coo]
    rated_bid = train_ratings[train_ratings['user_id'] == uid]['book_id'].values
    unrated_book_coo = np.where(~np.isin(book_ids, rated_bid))[0]
    unrated_predict = ratings[unrated_book_coo]
    recommend_book_coo = np.argsort(unrated_predict)[::-1][:100]
    np.save('./recommend/content-based/{}.npy'.format(uid), book_ids[unrated_book_coo][recommend_book_coo])