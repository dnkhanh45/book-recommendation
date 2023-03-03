import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('./data/train_ratings.csv')
test_df = pd.read_csv('./data/test_ratings.csv')
encode_books = np.load('./data/encode_books.npy')

user_ids = train_df['user_id'].unique()
user_ids.sort()

seed = 3973

for alpha in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    os.makedirs('./result/content-based-{}'.format(alpha), exist_ok=True)
    for uid in tqdm(user_ids[:], desc='alpha = {}'.format(alpha)):
        train_bids = train_df[train_df['user_id'] == uid]['book_id'].to_list()
        train_y = train_df[train_df['user_id'] == uid]['rating'].values.astype(np.float32)
        train_X = encode_books[[bid - 1 for bid in train_bids]]
        clf = Ridge(alpha=alpha, random_state=seed)
        clf.fit(train_X, train_y)
        uid_test_df = test_df[test_df['user_id'] == uid]
        test_bids = uid_test_df['book_id'].to_list()
        test_X = encode_books[[bid - 1 for bid in test_bids]]
        uid_test_df['predict'] = clf.predict(test_X)
        uid_test_df.to_csv('./result/content-based-{}/{}.csv'.format(alpha, uid), index=False)
