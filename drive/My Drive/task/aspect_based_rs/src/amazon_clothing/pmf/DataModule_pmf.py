import torch

import numpy as np 
from collections import defaultdict

import torch.utils.data as data

from copy import deepcopy

import config_pmf as conf

PAD = 0; SOS = 1; EOS = 2

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

def load_all():
    max_user, max_item = 0, 0
    train_data = {}
    f = open(train_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
        train_data[idx] = [user, item, rating]
        max_user = max(user, max_user)
        max_item = max(item, max_item)

    val_data = {}
    f = open(val_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
        val_data[idx] = [user, item, rating]
        max_user = max(user, max_user)
        max_item = max(item, max_item)
    
    test_data = {}
    f = open(test_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
        test_data[idx] = [user, item, rating]
        max_user = max(user, max_user)
        max_item = max(item, max_item)
    
    print('max_user:%d, max_item:%d' % (max_user, max_item))

    return train_data, val_data, test_data
        
class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        for idx in batch_idx_list:
            user_list.append(self.train_data[idx][0])
            item_list.append(self.train_data[idx][1])
            rating_list.append(self.train_data[idx][2])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda()
