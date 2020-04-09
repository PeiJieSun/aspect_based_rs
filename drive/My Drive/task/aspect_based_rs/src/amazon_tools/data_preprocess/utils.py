from collections import defaultdict
import numpy as np

import config as conf

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

user_reviews_dict, item_reviews_dict = defaultdict(list), defaultdict(list)
max_user, max_item = 0, 0
total_rating_list = []
f = open(train_data_path)
for line in f:
    line = eval(line)
    idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
    total_rating_list.append(rating)
    max_user = max(max_user, user)
    max_item = max(max_item, item)
    user_reviews_dict[user].append(idx)
    item_reviews_dict[item].append(idx)

f = open(val_data_path)
for line in f:
    line = eval(line)
    idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']

f = open(test_data_path)
for line in f:
    line = eval(line)
    idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']

from gensim.models import Word2Vec
word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))

P_REVIEW = 0.85

user_len_list = []
for user in user_reviews_dict:
    user_len_list.append(len(user_reviews_dict[user]))
x = np.sort(user_len_list)
xLen = len(x)
max_user_review_num = x[int(P_REVIEW * xLen) - 1]

item_len_list = []
for item in item_reviews_dict:
    item_len_list.append(len(item_reviews_dict[item]))
x = np.sort(item_len_list)
xLen = len(x)
max_item_review_num = x[int(P_REVIEW * xLen) - 1]

print('avg rating:%.4f' % (np.mean(total_rating_list)))
print('num users:%d' % (max_user + 1))
print('num items:%d' % (max_item + 1))
print('vocab sz:%d' % (len(word_embedding.wv.vocab)))
print('u_max_r:%d' % max_user_review_num)
print('i_max_r:%d' % max_item_review_num)