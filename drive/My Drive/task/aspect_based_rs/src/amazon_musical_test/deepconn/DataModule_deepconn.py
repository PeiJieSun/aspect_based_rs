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

def load_all():
    user_doc_dict, item_doc_dict = defaultdict(list), defaultdict(list)
    user_review_dict, item_review_dict = defaultdict(list), defaultdict(list)

    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, abae_review = \
            line['user'], line['item'], line['rating'], line['abae_review']

        train_data[idx] = [user, item, rating]

        for sent in abae_review:
            sent = sent[:conf.seq_len]
            sent.extend([PAD]*(conf.seq_len - len(sent)))
            user_doc_dict[user].append(sent)
            item_doc_dict[item].append(sent)

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

    for user in user_doc_dict:
        if len(user_doc_dict[user]) < conf.user_seq_num:
            user_doc_dict[user].extend([[PAD]*conf.seq_len]*(conf.user_seq_num-len(user_doc_dict[user])))
        else:
            user_doc_dict[user] = user_doc_dict[user][:conf.user_seq_num]

    for item in item_doc_dict:
        if len(item_doc_dict[item]) < conf.item_seq_num:
            item_doc_dict[item].extend([[PAD]*conf.seq_len]*(conf.item_seq_num-len(item_doc_dict[item])))
        else:
            item_doc_dict[item] = item_doc_dict[item][:conf.item_seq_num]

    for user, values in user_doc_dict.items():
        doc = []
        for value in values:
            doc.extend(value)
            user_doc_dict[user] = doc
            
    for item, values in item_doc_dict.items():
        doc = []
        for value in values:
            doc.extend(value)
            item_doc_dict[item] = doc
    
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
            user, item = self.train_data[data_idx][0], self.train_data[data_idx][1]
            user_list.append(user)
            item_list.append(item)
            rating_list.append(self.train_data[data_idx][2])
            
            user_doc.append(self.user_doc_dict[user])
            item_doc.append(self.item_doc_dict[item])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.array(user_doc)).cuda(),\
        torch.LongTensor(np.array(item_doc)).cuda()