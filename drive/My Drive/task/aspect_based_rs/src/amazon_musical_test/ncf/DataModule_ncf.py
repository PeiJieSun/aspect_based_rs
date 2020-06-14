import torch

import numpy as np 
from collections import defaultdict

import torch.utils.data as data

from copy import deepcopy

import config_ncf as conf

PAD = 0; SOS = 1; EOS = 2

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

def load_all():
    train_rating_list = []

    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating = line['user'], line['item'], line['rating']
        train_data[idx] = [user, item, rating]

        train_rating_list.append(rating)
    avg_rating = np.mean(train_rating_list)


    val_rating_list = []
    val_data = {}
    f = open(val_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating = line['user'], line['item'], line['rating']
        val_data[idx] = [user, item, rating]

        val_rating_list.append(rating)

    test_rating_list = []
    test_data = {}
    f = open(test_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating = line['user'], line['item'], line['rating']
        test_data[idx] = [user, item, rating]

        test_rating_list.append(rating)
    
    print('Train RMSE:%.4f' % np.sqrt(np.mean((train_rating_list-avg_rating)**2)))
    print('Val RMSE:%.4f' % np.sqrt(np.mean((val_rating_list-avg_rating)**2)))
    print('Test RMSE:%.4f' % np.sqrt(np.mean((test_rating_list-avg_rating)**2)))

    print('avg rating:%.4f' % avg_rating)
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
