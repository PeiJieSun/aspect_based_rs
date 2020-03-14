import gzip
import json

from gensim.models import Word2Vec

import numpy as np
from collections import defaultdict

data_name = 'amazon_sports'
file_path = '/content/drive/My Drive/datasets/amazon_sports/'
origin_file = '%s/reviews_Sports_and_Outdoors_5.json.gz' % file_path

'''
train_user_historical_review_dict, train_item_historical_review_dict = defaultdict(list), defaultdict(list)

g = gzip.open(origin_file, 'r')
for idx, line in enumerate(g):
    line = eval(line)
    user, item = line['reviewerID'], line['asin']
    train_user_historical_review_dict[user].append(idx)
    train_item_historical_review_dict[item].append(idx)

user_consumption_list = []
item_consumption_list = []
for user, consumption_list in train_user_historical_review_dict.items():
    user_consumption_list.append(len(consumption_list))
for item, consumption_list in train_item_historical_review_dict.items():
    item_consumption_list.append(len(consumption_list))

user_consumption_list.sort(reverse=True)
item_consumption_list.sort(reverse=True)

import pdb; pdb.set_trace()
'''

data_name = 'amazon_sports'
target_path = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_sports'
train_data_path = '%s/%s.train.data' % (target_path, data_name)

user_set, item_set, rating_list = set(), set(), []
train_data = {}
f = open(train_data_path)
for line in f:
    line = eval(line)
    idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
    user_set.add(user)
    item_set.add(item)
    rating_list.append(rating)

word_embedding = Word2Vec.load('/content/drive/My Drive/task/aspect_based_rs/data/amazon_sports/amazon_sports.wv.model')

print('num of users:%d, num of items:%d, vocab size:%d, average rating:%.4f' % (max(user_set)+1, max(item_set)+1, len(word_embedding.wv.vocab)+3, np.mean(rating_list)))