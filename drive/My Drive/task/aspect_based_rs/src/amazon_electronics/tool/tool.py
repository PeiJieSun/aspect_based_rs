import numpy as np 
from collections import defaultdict


data_name = 'amazon_electronics'
target_path = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_electronics'
train_data_path = '%s/%s.train.data' % (target_path, data_name)
val_data_path = '%s/%s.val.data' % (target_path, data_name)
test_data_path = '%s/%s.test.data' % (target_path, data_name)

tain_review_embedding_path = '%s/%s.train.review_embedding.npy' % (target_path, data_name)
val_review_embedding_path = '%s/%s.val.review_embedding.npy' % (target_path, data_name)
test_review_embedding_path = '%s/%s.test.review_embedding.npy' % (target_path, data_name)

train_user_historical_review_dict, train_item_historical_review_dict = defaultdict(list), defaultdict(list)

train_data = {}
f = open(train_data_path)
for line in f:
    line = eval(line)
    idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
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