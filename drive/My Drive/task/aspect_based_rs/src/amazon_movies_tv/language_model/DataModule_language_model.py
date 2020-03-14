import torch

import numpy as np 
from collections import defaultdict

import torch.utils.data as data
import config_language_model as conf

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

data_name = 'amazon_movies_tv'
target_path = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_movies_tv'
train_data_path = '%s/%s.train.data' % (target_path, data_name)
val_data_path = '%s/%s.val.data' % (target_path, data_name)
test_data_path = '%s/%s.test.data' % (target_path, data_name)

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
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in, review_out = generate_review(review)
        train_data[idx] = [user, item, rating, review_in, review_out]

    val_data = {}
    f = open(val_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in, review_out = generate_review(review)
        val_data[idx] = [user, item, rating, review_in, review_out]
    
    test_data = {}
    f = open(test_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in, review_out = generate_review(review)
        test_data[idx] = [user, item, rating, review_in, review_out]
        
    return train_data, val_data, test_data
        
class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list, review_input_list, review_output_list = [], [], [], [], []
        for idx in batch_idx_list:
            user_list.append(self.train_data[idx][0])
            item_list.append(self.train_data[idx][1])
            rating_list.append(self.train_data[idx][2])
            review_input_list.append(self.train_data[idx][3])
            review_output_list.append(self.train_data[idx][4])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(review_output_list)).cuda()