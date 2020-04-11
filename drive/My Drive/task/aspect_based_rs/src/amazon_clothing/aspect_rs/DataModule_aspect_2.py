# ABAE + DeepCoNN

import torch

import numpy as np 
from collections import defaultdict
from collections import OrderedDict

import torch.utils.data as data
import config_aspect_2 as conf

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2
MAX_DOC_LEN = 400

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

def generate_review(review):
    review_in = [SOS]
    review_in.extend(review)
    return review_in

def load_all():
    user_doc_dict, item_doc_dict = defaultdict(list), defaultdict(list)
    train_user_historical_review_dict, train_item_historical_review_dict = defaultdict(list), defaultdict(list)

    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, review = line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        train_data[idx] = [user, item, rating, review_in]
        if len(train_user_historical_review_dict[user]) < conf.u_max_r:
            user_doc_dict[user].extend(review)
            train_user_historical_review_dict[user].append(idx)
        if len(train_item_historical_review_dict[item]) < conf.i_max_r:
            item_doc_dict[item].extend(review)
            train_item_historical_review_dict[item].append(idx)

    for user, doc in user_doc_dict.items():
        if len(doc) > MAX_DOC_LEN:
            user_doc_dict[user] = doc[:MAX_DOC_LEN]
        else:
            user_doc_dict[user].extend([PAD]*(MAX_DOC_LEN-len(doc)))

    for item, doc in item_doc_dict.items():
        if len(doc) > MAX_DOC_LEN:
            item_doc_dict[item] = doc[:MAX_DOC_LEN]
        else:
            item_doc_dict[item].extend([PAD]*(MAX_DOC_LEN-len(doc)))

    val_data = {}
    f = open(val_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating = line['user'], line['item'], line['rating']
        val_data[idx] = [user, item, rating]
    
    test_data = {}
    f = open(test_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating = line['user'], line['item'], line['rating']
        review_in = generate_review(review)
        test_data[idx] = [user, item, rating, review_in]
    
    #import pdb; pdb.set_trace()
    return train_data, val_data, test_data, \
        train_user_historical_review_dict, train_item_historical_review_dict, user_doc_dict, item_doc_dict

class TrainData():
    def __init__(self, train_data, 
        user_historical_review_dict, 
        item_historical_review_dict, review_data,
        user_doc_dict, item_doc_dict):
        self.review_data = review_data
        self.train_data = train_data
        self.length = len(train_data.keys())
        self.user_historical_review_dict = user_historical_review_dict
        self.item_historical_review_dict = item_historical_review_dict
        self.user_doc_dict = user_doc_dict
        self.item_doc_dict = item_doc_dict

    def get_batch(self, batch_idx_list):
        old_user_histor_idx, old_item_histor_idx = OrderedDict(), OrderedDict()
        review_idx_set = set()
        user_list, item_list, rating_list = [], [], []
        user_doc, item_doc = [], []
        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0])
            item_list.append(self.train_data[data_idx][1])
            rating_list.append(self.train_data[data_idx][2])
            
            user, item = self.train_data[data_idx][0], self.train_data[data_idx][1]

            # Prepare data for DeepCoNN Model
            if user in self.user_doc_dict:
                user_doc.append(self.user_doc_dict[self.train_data[data_idx][0]])
            else:
                user_doc.append([PAD]*MAX_DOC_LEN)
            if item in self.item_doc_dict:
                item_doc.append(self.item_doc_dict[self.train_data[data_idx][1]])
            else:
                item_doc.append([PAD]*MAX_DOC_LEN)

            # Prepare data for ABAE Model
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

        user_histor_index_l1, user_histor_index_l2, user_histor_value = [], [], []
        item_histor_index_l1, item_histor_index_l2, item_histor_value = [], [], []

        for idx, key in enumerate(old_user_histor_idx):
            old_review_idx_list = old_user_histor_idx[key]
            for review_idx in old_review_idx_list:
                user_histor_index_l1.append(idx)
                user_histor_index_l2.append(review_mapping[review_idx])
                user_histor_value.append(1.0/len(old_review_idx_list))
        for idx, key in enumerate(old_item_histor_idx):
            old_review_idx_list = old_item_histor_idx[key]
            for review_idx in old_review_idx_list:
                item_histor_index_l1.append(idx)
                item_histor_index_l2.append(review_mapping[review_idx])
                item_histor_value.append(1.0/len(old_review_idx_list))
        user_histor_index = [user_histor_index_l1, user_histor_index_l2]
        item_histor_index = [item_histor_index_l1, item_histor_index_l2]
        
        # prepare review positive/negative embedding, and review words input
        neg_review = []
        review_input_list = []
        for review_idx in review_idx_list:
            review_input_list.append(self.review_data[review_idx][3])
            for _ in range(conf.num_negative_reviews):
                j = np.random.randint(self.length-1)
                while j == idx:
                    j = np.random.randint(self.length-1)
                neg_review.append(self.review_data[j][3])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.array(review_input_list)).cuda(),\
        torch.LongTensor(np.array(neg_review)).cuda(),\
        torch.LongTensor(user_histor_index).cuda(),\
        torch.FloatTensor(user_histor_value).cuda(),\
        torch.LongTensor(item_histor_index).cuda(),\
        torch.FloatTensor(item_histor_value).cuda(),\
        torch.LongTensor(np.array(user_doc)).cuda(),\
        torch.LongTensor(np.array(item_doc)).cuda()