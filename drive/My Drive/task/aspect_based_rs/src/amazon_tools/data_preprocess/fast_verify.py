import gzip
import json

import numpy as np

import config as conf 

from copy import deepcopy


def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

check_dir('%s/%s.train.rating' % (conf.target_path, conf.data_name))
train_data = open('%s/%s.train.rating' % (conf.target_path, conf.data_name), 'w')
val_data = open('%s/%s.val.rating' % (conf.target_path, conf.data_name), 'w')
test_data = open('%s/%s.test.rating' % (conf.target_path, conf.data_name), 'w')

# first is generate the train / val / test data, then write them to the files.
train_data_list, test_data_list = [], []
train_user, train_item = (), ()

user_idx_dict, item_idx_dict = {}, {}

g = gzip.open(conf.origin_file, 'r')
for idx, line in enumerate(g):
    line = eval(line)
    if (idx+1) % 5 == 0:
        test_data_list.append([line['reviewerID'], line['asin'], line['overall'], line['reviewText']])
    else:
        if line['reviewerID'] not in user_idx_dict:
            user_idx_dict[line['reviewerID']] = len(user_idx_dict.keys())
        if line['asin'] not in item_idx_dict:
            item_idx_dict[line['asin']] = len(item_idx_dict.keys())
        train_data_list.append([line['reviewerID'], line['asin'], line['overall'], line['reviewText']])
print('data read complete.')

# write train data set to file
idx = 0
train_review_embedding = {}
for idx, value in enumerate(train_data_list):
    user, item, rating = user_idx_dict[value[0]], item_idx_dict[value[1]], value[2]
    if (idx+1) % 10 == 0:
        val_data.write('%d\t%d\t%d\n' % (user, item, rating))
    else:
        train_data.write('%d\t%d\t%d\n' % (user, item, rating))

# write test data set to file
idx = 0
for value in test_data_list:
    if value[0] in user_idx_dict and value[1] in item_idx_dict:
        user, item, rating = user_idx_dict[value[0]], item_idx_dict[value[1]], value[2]
        test_data.write('%d\t%d\t%d\n' % (user, item, rating))