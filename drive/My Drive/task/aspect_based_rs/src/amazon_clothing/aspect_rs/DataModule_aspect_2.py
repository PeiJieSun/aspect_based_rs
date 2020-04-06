import torch

import numpy as np 
from collections import defaultdict
from collections import OrderedDict

import torch.utils.data as data
import config_aspect_2 as conf

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

def generate_review(review):
    review_in = [SOS]
    review_in.extend(review)
    return review_in

def load_all():
    train_user_historical_review_dict, train_item_historical_review_dict = defaultdict(list), defaultdict(list)

    train_data = {}
    f = open(train_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review = line['idx'], line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        train_data[idx] = [user, item, rating, review_in]
        if len(train_user_historical_review_dict[user]) < conf.u_max_r:
            train_user_historical_review_dict[user].append(idx)
        if len(train_item_historical_review_dict[item]) < conf.i_max_r:
            train_item_historical_review_dict[item].append(idx)

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
    
    #import pdb; pdb.set_trace()
    return train_data, val_data, test_data, train_user_historical_review_dict, train_item_historical_review_dict

class TrainData():
    def __init__(self, train_data, 
        user_historical_review_dict, 
        item_historical_review_dict):
        self.train_data = train_data
        self.length = len(train_data.keys())
        self.user_historical_review_dict = user_historical_review_dict
        self.item_historical_review_dict = item_historical_review_dict

    def get_batch(self, batch_idx_list):
        old_user_histor_idx, old_item_histor_idx = OrderedDict(), OrderedDict()
        user_review_idx_set, item_review_idx_set = set(), set()
        user_list, item_list, rating_list = [], [], []
        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0])
            item_list.append(self.train_data[data_idx][1])
            rating_list.append(self.train_data[data_idx][2])
            
            user, item = self.train_data[data_idx][0], self.train_data[data_idx][0]
            old_user_histor_idx[data_idx] = self.user_historical_review_dict[user]
            for review_idx in self.user_historical_review_dict[user]:
                user_review_idx_set.add(review_idx)
            old_item_histor_idx[data_idx] = self.item_historical_review_dict[item]
            for review_idx in self.item_historical_review_dict[item]:
                item_review_idx_set.add(review_idx)

        user_review_idx_list = list(user_review_idx_set)
        item_review_idx_list = list(item_review_idx_set)

        # rearrange all reviews in current batch
        user_review_mapping = {}
        for rearrange_idx, review_idx in enumerate(user_review_idx_list):
            user_review_mapping[review_idx] = rearrange_idx

        user_histor_index_l1, user_histor_index_l2, user_histor_value = [], [], []
        for idx, key in enumerate(old_user_histor_idx):
            old_review_idx_list = old_user_histor_idx[key]
            for review_idx in old_review_idx_list:
                user_histor_index_l1.append(idx)
                user_histor_index_l2.append(user_review_mapping[review_idx])
                user_histor_value.append(1.0/len(old_review_idx_list))
        user_histor_index = [user_histor_index_l1, user_histor_index_l2]
        
        # prepare review positive/negative embedding, and review words input
        user_neg_review = []
        user_review_input_list = []
        for review_idx in user_review_idx_list:
            user_review_input_list.append(self.train_data[review_idx][3])
            for _ in range(conf.num_negative_reviews):
                j = np.random.randint(self.length-1)
                while j == review_idx:
                    j = np.random.randint(self.length-1)
                user_neg_review.append(self.train_data[j][3])

        ##########################################################################
        # rearrange all reviews in current batch
        item_review_mapping = {}
        for rearrange_idx, review_idx in enumerate(item_review_idx_list):
            item_review_mapping[review_idx] = rearrange_idx

        item_histor_index_l1, item_histor_index_l2, item_histor_value = [], [], []
        for idx, key in enumerate(old_item_histor_idx):
            old_review_idx_list = old_item_histor_idx[key]
            for review_idx in old_review_idx_list:
                item_histor_index_l1.append(idx)
                item_histor_index_l2.append(item_review_mapping[review_idx])
                item_histor_value.append(1.0/len(old_review_idx_list))
        item_histor_index = [item_histor_index_l1, item_histor_index_l2]

        item_neg_review = []
        item_review_input_list = []
        for review_idx in item_review_idx_list:
            item_review_input_list.append(self.train_data[review_idx][3])
            for _ in range(conf.num_negative_reviews):
                j = np.random.randint(self.length-1)
                while j == review_idx:
                    j = np.random.randint(self.length-1)
                item_neg_review.append(self.train_data[j][3])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.array(user_review_input_list)).cuda(),\
        torch.LongTensor(np.array(user_neg_review)).cuda(),\
        torch.LongTensor(np.array(item_review_input_list)).cuda(),\
        torch.LongTensor(np.array(item_neg_review)).cuda(),\
        torch.LongTensor(user_histor_index).cuda(),\
        torch.FloatTensor(user_histor_value).cuda(),\
        torch.LongTensor(item_histor_index).cuda(),\
        torch.FloatTensor(item_histor_value).cuda()

class ValData():
    def __init__(self, val_data):
        self.val_data = val_data
        self.length = len(val_data.keys())

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        for data_idx in batch_idx_list:
            user_list.append(self.val_data[data_idx][0])
            item_list.append(self.val_data[data_idx][1])
            rating_list.append(self.val_data[data_idx][2])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda()