import torch
import torch.nn.functional as F

import numpy as np 
from collections import defaultdict

import torch.utils.data as data
import config_mrg as conf

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

def generate_review(review):
    review_in = [SOS]
    review_in.extend(review)
    review_out = review
    review_out.append(EOS)
    return review_in, review_out

def load_all():
    train_data = {}
    f = open(train_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, g_review = line['idx'], line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review)
        train_data[idx] = [user, item, rating, review_in, review_out]

    val_data = {}
    f = open(val_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, g_review = line['idx'], line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review)
        val_data[idx] = [user, item, rating, review_in, review_out]
    
    test_data = {}
    f = open(test_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, g_review = line['idx'], line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review)
        test_data[idx] = [user, item, rating, review_in, review_out]
        
    return train_data, val_data, test_data
    
class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

    def get_batch(self, batch_idx_list):        
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0])
            item_list.append(self.train_data[data_idx][1])
            rating_list.append(self.train_data[data_idx][2])

            review_input_list.append(self.train_data[data_idx][3])
            review_output_list.append(self.train_data[data_idx][4])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(review_output_list)).cuda()