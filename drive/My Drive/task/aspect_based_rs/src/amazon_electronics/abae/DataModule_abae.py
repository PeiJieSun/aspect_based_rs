import torch

import numpy as np 
from collections import defaultdict

import torch.utils.data as data
import config_abae as conf

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

data_name = 'amazon_electronics'
target_path = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_electronics'
train_data_path = '%s/%s.train.data' % (target_path, data_name)
val_data_path = '%s/%s.val.data' % (target_path, data_name)
test_data_path = '%s/%s.test.data' % (target_path, data_name)

tain_review_embedding_path = '%s/%s.train.review_embedding.npy' % (target_path, data_name)
val_review_embedding_path = '%s/%s.val.review_embedding.npy' % (target_path, data_name)
test_review_embedding_path = '%s/%s.test.review_embedding.npy' % (target_path, data_name)

def generate_review(review):
    review_in = [SOS]
    review_in.extend(review)
    return review_in

def load_all():
    train_data = {}
    f = open(train_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        train_data[idx] = [user, item, rating, review_in]

    val_data = {}
    f = open(val_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        val_data[idx] = [user, item, rating, review_in]
    
    test_data = {}
    f = open(test_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        test_data[idx] = [user, item, rating, review_in]
    
    train_review_embedding = np.load(tain_review_embedding_path, allow_pickle=True).item()
    val_review_embedding = np.load(val_review_embedding_path, allow_pickle=True).item()
    test_review_embedding = np.load(test_review_embedding_path, allow_pickle=True).item()

    return train_data, val_data, test_data, train_review_embedding, val_review_embedding, test_review_embedding
        
class TrainData():
    def __init__(self, train_data, review_embedding_dict):
        self.train_data = train_data
        self.length = len(train_data.keys())
        self.review_embedding_dict = review_embedding_dict

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list, review_input_list = [], [], [], []
        review_pos_embedding, review_neg_embedding = [], []
        for idx in batch_idx_list:
            user_list.append(self.train_data[idx][0])
            item_list.append(self.train_data[idx][1])
            rating_list.append(self.train_data[idx][2])
            review_input_list.append(self.train_data[idx][3])

            review_pos_embedding.append(self.review_embedding_dict[idx])
        
            for _ in range(conf.num_negative_reviews):
                j = np.random.randint(self.length-1)
                while j == idx:
                    j = np.random.randint(self.length-1)
                review_neg_embedding.append(self.review_embedding_dict[j])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.array(review_input_list)).cuda(),\
        torch.FloatTensor(review_pos_embedding).cuda(),\
        torch.FloatTensor(review_neg_embedding).cuda()
