import torch

import numpy as np 
from collections import defaultdict
from collections import OrderedDict

import torch.utils.data as data
import config_aspect as conf

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
    train_user_historical_review_dict, train_item_historical_review_dict = defaultdict(list), defaultdict(list)
    val_user_historical_review_dict, val_item_historical_review_dict = defaultdict(list), defaultdict(list)
    test_user_historical_review_dict, test_item_historical_review_dict = defaultdict(list), defaultdict(list)

    train_data = {}
    f = open(train_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        train_data[idx] = [user, item, rating, review_in]
        train_user_historical_review_dict[user].append(idx)
        train_item_historical_review_dict[item].append(idx)

        if idx > 100000:
            break

    val_data = {}
    f = open(val_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        val_data[idx] = [user, item, rating, review_in]
        val_user_historical_review_dict[user].append(idx)
        val_item_historical_review_dict[item].append(idx)

        if idx > 10000:
            break
    
    test_data = {}
    f = open(test_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        test_data[idx] = [user, item, rating, review_in]
        test_user_historical_review_dict[user].append(idx)
        test_item_historical_review_dict[item].append(idx)

        if idx > 10000:
            break
    
    train_review_embedding = np.load(tain_review_embedding_path, allow_pickle=True).item()
    val_review_embedding = np.load(val_review_embedding_path, allow_pickle=True).item()
    test_review_embedding = np.load(test_review_embedding_path, allow_pickle=True).item()

    return train_data, val_data, test_data, train_review_embedding, val_review_embedding, test_review_embedding,\
    train_user_historical_review_dict, train_item_historical_review_dict,\
    val_user_historical_review_dict, val_item_historical_review_dict,\
    test_user_historical_review_dict, test_item_historical_review_dict

'''
class TrainData():
    def __init__(self, train_data, review_embedding_dict, 
        user_historical_review_dict, 
        item_historical_review_dict):
        self.train_data = train_data
        self.length = len(train_data.keys())
        self.review_embedding_dict = review_embedding_dict
        self.user_historical_review_dict = user_historical_review_dict
        self.item_historical_review_dict = item_historical_review_dict

    def get_batch(self, batch_idx_list):
        old_user_histor_idx, old_item_histor_idx = OrderedDict(), OrderedDict()
        review_idx_set = set()
        user_list, item_list, rating_list = [], [], []
        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0])
            item_list.append(self.train_data[data_idx][1])
            rating_list.append(self.train_data[data_idx][2])
            
            user, item = self.train_data[data_idx][0], self.train_data[data_idx][1]
            old_user_histor_idx[data_idx] = self.user_historical_review_dict[user]
            old_item_histor_idx[data_idx] = self.item_historical_review_dict[item]

            for review_idx in self.user_historical_review_dict[user]:
                review_idx_set.add(review_idx)
            for review_idx in self.item_historical_review_dict[item]:
                review_idx_set.add(review_idx)

        review_idx_list = list(review_idx_set)
        # rearrange all reviews in current batch
        review_mapping = {}
        for rearrange_idx, review_idx in enumerate(review_idx_list):
            review_mapping[review_idx] = rearrange_idx

        user_histor_idx, item_histor_idx = OrderedDict(), OrderedDict()
        # rearrange all reviews in current batch
        for idx, key in enumerate(old_user_histor_idx):
            old_review_idx_list = old_user_histor_idx[key]
            tmp_review_idx_list = []
            for old_review_idx in old_review_idx_list:
                tmp_review_idx_list.append(review_mapping[old_review_idx])
            user_histor_idx[idx] = tmp_review_idx_list
        for idx, key in enumerate(old_item_histor_idx):
            old_review_idx_list = old_item_histor_idx[key]
            tmp_review_idx_list = []
            for old_review_idx in old_review_idx_list:
                tmp_review_idx_list.append(review_mapping[old_review_idx])
            item_histor_idx[idx] = tmp_review_idx_list

        # prepare review positive/negative embedding, and review words input
        review_pos_embedding, review_neg_embedding = [], []
        review_input_list = []
        for review_idx in review_idx_list:
            review_input_list.append(self.train_data[review_idx][3])
            review_pos_embedding.append(self.review_embedding_dict[review_idx])
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
        torch.FloatTensor(review_neg_embedding).cuda(),\
        user_histor_idx, item_histor_idx
'''

class TrainData():
    def __init__(self, train_data, review_embedding_dict, 
        user_historical_review_dict, 
        item_historical_review_dict):
        self.train_data = train_data
        self.review_embedding_dict = review_embedding_dict
        self.user_historical_review_dict = user_historical_review_dict
        self.item_historical_review_dict = item_historical_review_dict
        self.item_list = list(item_historical_review_dict.keys())
        self.length = len(self.item_list)

    def get_batch(self, batch_item_list):
        batch_idx_list = set()
        for item in batch_item_list:
            review_idx_list = self.item_historical_review_dict[item]
            for review_idx in review_idx_list:
                batch_idx_list.add(review_idx)
        batch_idx_list = list(batch_idx_list)
        
        old_user_histor_idx, old_item_histor_idx = OrderedDict(), OrderedDict()
        review_idx_set = set()
        user_list, item_list, rating_list = [], [], []
        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0])
            item_list.append(self.train_data[data_idx][1])
            rating_list.append(self.train_data[data_idx][2])
            
            user, item = self.train_data[data_idx][0], self.train_data[data_idx][1]
            old_user_histor_idx[data_idx] = self.user_historical_review_dict[user]
            old_item_histor_idx[data_idx] = self.item_historical_review_dict[item]

            for review_idx in self.user_historical_review_dict[user]:
                review_idx_set.add(review_idx)
            for review_idx in self.item_historical_review_dict[item]:
                review_idx_set.add(review_idx)

        review_idx_list = list(review_idx_set)
        # rearrange all reviews in current batch
        review_mapping = {}
        for rearrange_idx, review_idx in enumerate(review_idx_list):
            review_mapping[review_idx] = rearrange_idx

        user_histor_idx, item_histor_idx = OrderedDict(), OrderedDict()
        # rearrange all reviews in current batch
        for idx, key in enumerate(old_user_histor_idx):
            old_review_idx_list = old_user_histor_idx[key]
            tmp_review_idx_list = []
            for old_review_idx in old_review_idx_list:
                tmp_review_idx_list.append(review_mapping[old_review_idx])
            user_histor_idx[idx] = tmp_review_idx_list
        for idx, key in enumerate(old_item_histor_idx):
            old_review_idx_list = old_item_histor_idx[key]
            tmp_review_idx_list = []
            for old_review_idx in old_review_idx_list:
                tmp_review_idx_list.append(review_mapping[old_review_idx])
            item_histor_idx[idx] = tmp_review_idx_list

        # prepare review positive/negative embedding, and review words input
        review_pos_embedding, review_neg_embedding = [], []
        review_input_list = []
        for review_idx in review_idx_list:
            review_input_list.append(self.train_data[review_idx][3])
            review_pos_embedding.append(self.review_embedding_dict[review_idx])
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
        torch.FloatTensor(review_neg_embedding).cuda(),\
        user_histor_idx, item_histor_idx