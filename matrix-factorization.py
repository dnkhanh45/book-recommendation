import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import argparse
tqdm.pandas()
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('dim', type=int)
dim = parser.parse_args().dim
user_w = torch.zeros(user_ids.shape[0], dim, requires_grad=True)
user_b = torch.zeros(user_ids.shape[0], requires_grad=True)
book_w = torch.zeros(book_ids.shape[0], dim, requires_grad=True)
book_b = torch.zeros(book_ids.shape[0], requires_grad=True)

seed = 3973
torch.manual_seed(seed=seed)
torch.nn.init.xavier_uniform_(user_w)
torch.nn.init.xavier_uniform_(book_w)

user_w = user_w.to(device)
user_b = user_b.to(device)
book_w = book_w.to(device)
book_b = book_b.to(device)

mu = train_df['rating'].mean()
lambda_ = 0.01
lr = 0.5


os.makedirs('./learn/matrix-factorization-{}'.format(dim), exist_ok=True)
with open('./learn/matrix-factorization-{}/log.csv'.format(dim), 'w') as f:
    f.write('train_mse,test_mse,train,test\n')

for epoch in range(50):
    train_mse_loss = 0.0
    for index in tqdm(range(train_df.shape[0]), desc='train: dim={}, epoch={}'.format(dim, epoch + 1)):
        row = train_df.iloc[index]
        user_coo = uidd[row['user_id']]
        book_coo = bidd[row['book_id']]
        train_mse_loss += (
            user_w[user_coo].dot(book_w[book_coo]) + \
            user_b[user_coo] + book_b[book_coo] + \
            mu - row['rating']
        ) ** 2
    train_mse_loss = train_mse_loss / train_df.shape[0] / 2
    train_loss = train_mse_loss + lambda_ * (
        torch.pow(user_w, 2).sum() + \
        torch.pow(user_b, 2).sum() + \
        torch.pow(book_w, 2).sum() + \
        torch.pow(book_b, 2).sum()
    )
    user_w.retain_grad()
    user_b.retain_grad()
    book_w.retain_grad()
    book_b.retain_grad()
    train_loss.backward()
    with torch.no_grad():
        user_w = user_w - lr * user_w.grad
        user_b = user_b - lr * user_b.grad
        book_w = book_w - lr * book_w.grad
        book_b = book_b - lr * book_b.grad
        user_w.grad = None
        user_b.grad = None
        book_w.grad = None
        book_b.grad = None

        test_mse_loss = 0.0
        for index in tqdm(range(test_df.shape[0]), desc='test: dim={}, epoch={}'.format(dim, epoch + 1)):
            row = train_df.iloc[index]
            user_coo = uidd[row['user_id']]
            book_coo = bidd[row['book_id']]
            test_mse_loss += (
                user_w[user_coo].dot(book_w[book_coo]) + \
                user_b[user_coo] + book_b[book_coo] + \
                mu - row['rating']
            ) ** 2
        test_mse_loss = test_mse_loss / test_df.shape[0] / 2
        test_loss = test_mse_loss + lambda_ * (
            torch.pow(user_w, 2).sum() + \
            torch.pow(user_b, 2).sum() + \
            torch.pow(book_w, 2).sum() + \
            torch.pow(book_b, 2).sum()
        )
    user_w.requires_grad = True
    user_b.requires_grad = True
    book_w.requires_grad = True
    book_b.requires_grad = True
    with open('./learn/matrix-factorization-{}/log.csv'.format(dim), 'a') as f:
        f.write('{},{},{},{}\n'.format(train_mse_loss.item(), test_mse_loss.item(), train_loss.item(), test_loss.item()))

np.save('./learn/matrix-factorization-{}/user_w.npy'.format(dim), user_w.detach().cpu().numpy())
np.save('./learn/matrix-factorization-{}/user_b.npy'.format(dim), user_b.detach().cpu().numpy())
np.save('./learn/matrix-factorization-{}/book_w.npy'.format(dim), book_w.detach().cpu().numpy())
np.save('./learn/matrix-factorization-{}/book_b.npy'.format(dim), book_b.detach().cpu().numpy())
    