import torch

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
    review_in = review[:-1]
    review_in.extend([PAD]*(conf.rev_len-len(review_in)))
    review_out = review[1:]
    review_out.extend([PAD]*(conf.rev_len-len(review_out)))
    return review_in, review_out

def load_all():
    first_word_dict = defaultdict(int)

    max_user, max_item = 0, 0
    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review = line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review[:conf.rev_len])
        train_data[idx] = [user, item, rating, review_in, review_out, g_review[:conf.rev_len]]

        max_user = max(max_user, user)
        max_item = max(max_item, item)

    val_data = {}
    f = open(val_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review = line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review[:conf.rev_len])
        val_data[idx] = [user, item, rating, review_in, review_out, g_review[:conf.rev_len]]
    
    test_data = {}
    f = open(test_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review = line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review[:conf.rev_len])
        test_data[idx] = [user, item, rating, review_in, review_out, g_review[:conf.rev_len]]
    
    #import pdb; pdb.set_trace()
    return train_data, val_data, test_data
    
class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list = [], []
        review_aspect_bool_list, review_aspect_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0]) # (batch_size, 1)
            item_list.append(self.train_data[data_idx][1]) # (batch_size, 1)
            rating_list.append(self.train_data[data_idx][2]) # (batch_size, 1)

            review_input_list.append(self.train_data[data_idx][3]) #(batch_size, seq_length)
            review_output_list.append(self.train_data[data_idx][4]) #(batch_size, seq_length)
        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(review_output_list)).cuda()

class TestData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())
    
    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list, real_review_list = [], [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0]) # (batch_size, 1)
            item_list.append(self.train_data[data_idx][1]) # (batch_size, 1)
            rating_list.append(self.train_data[data_idx][2]) # (batch_size, 1)

            review_input_list.append(self.train_data[data_idx][3]) #(batch_size, seq_length)
            real_review_list.append(self.train_data[data_idx][5]) #(batch_size, seq_length) real review without PAD

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.LongTensor(review_input_list).cuda(), \
        torch.LongTensor(real_review_list).cuda()