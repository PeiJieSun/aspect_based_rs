import torch

import numpy as np 
from collections import defaultdict
from collections import OrderedDict

import torch.utils.data as data
import config_deepconn as conf

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
    user_review_dict, item_review_dict = defaultdict(list), defaultdict(list)

    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, review = line['user'], line['item'], line['rating'], line['review']
        review_in = generate_review(review)
        train_data[idx] = [user, item, rating, review_in]

        if len(user_review_dict[user]) < conf.u_max_r:
            user_doc_dict[user].extend(review)
            user_review_dict[user].append(idx)
        if len(item_review_dict[item]) < conf.i_max_r:
            item_doc_dict[item].extend(review)
            item_review_dict[item].append(idx)

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
        test_data[idx] = [user, item, rating]
    
    #import pdb; pdb.set_trace()
    return train_data, val_data, test_data, user_doc_dict, item_doc_dict

class TrainData():
    def __init__(self, train_data, user_doc_dict, item_doc_dict):
        self.train_data = train_data
        self.length = len(train_data.keys())
        self.user_doc_dict = user_doc_dict
        self.item_doc_dict = item_doc_dict

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        user_doc, item_doc = [], []
        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0])
            item_list.append(self.train_data[data_idx][1])
            rating_list.append(self.train_data[data_idx][2])
            
            user, item = self.train_data[data_idx][0], self.train_data[data_idx][1]
            if user in self.user_doc_dict:
                user_doc.append(self.user_doc_dict[self.train_data[data_idx][0]])
            else:
                user_doc.append([PAD]*MAX_DOC_LEN)
            if item in self.item_doc_dict:
                item_doc.append(self.item_doc_dict[self.train_data[data_idx][1]])
            else:
                item_doc.append([PAD]*MAX_DOC_LEN)

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.array(user_doc)).cuda(),\
        torch.LongTensor(np.array(item_doc)).cuda()

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