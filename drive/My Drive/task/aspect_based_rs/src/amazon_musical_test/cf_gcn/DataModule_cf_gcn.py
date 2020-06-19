import torch

import numpy as np 
from collections import defaultdict

import torch.utils.data as data
import config_cf_gcn as conf

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

def generate_review(review):
    review_in = review[:-1]
    review_in.extend([PAD]*(30-len(review_in)))
    review_out = review[1:]
    review_out.extend([PAD]*(30-len(review_out)))
    return review_in, review_out

def load_all():
    user_doc_dict, item_doc_dict = defaultdict(list), defaultdict(list)

    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review, abae_review = line['user'], line['item'],\
            line['rating'], line['g_review'], line['abae_review']
        review_in, review_out = generate_review(g_review[:conf.rev_len])

        g_review = g_review[:conf.rev_len]
        g_review.extend([PAD]*(conf.rev_len-len(g_review)))
        train_data[idx] = [user, item, rating, review_in, review_out, g_review]

        for sent in abae_review:
            sent = sent[:conf.seq_len]
            sent.extend([PAD]*(conf.seq_len - len(sent)))
            user_doc_dict[user].append(sent)
            item_doc_dict[item].append(sent)

    val_data = {}
    f = open(val_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review = \
            line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review[:conf.rev_len])

        g_review = g_review[:conf.rev_len]
        g_review.extend([PAD]*(conf.rev_len-len(g_review)))
        val_data[idx] = [user, item, rating, review_in, review_out, g_review]
    
    test_data = {}
    f = open(test_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review = \
            line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review[:conf.rev_len])

        g_review = g_review[:conf.rev_len]
        g_review.extend([PAD]*(conf.rev_len-len(g_review)))
        test_data[idx] = [user, item, rating, review_in, review_out, g_review]
    
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

    #import pdb; pdb.set_trace()
    return train_data, val_data, test_data, user_doc_dict, item_doc_dict
    
class TrainData():
    def __init__(self, train_data, user_doc_dict=None, item_doc_dict=None):
        self.train_data = train_data
        self.length = len(train_data.keys())
        self.user_doc_dict = user_doc_dict
        self.item_doc_dict = item_doc_dict

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list = [], []
        review_aspect_bool_list, review_aspect_list = [], []

        user_doc_list, item_doc_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0]) # (batch_size, 1)
            item_list.append(self.train_data[data_idx][1]) # (batch_size, 1)
            rating_list.append(self.train_data[data_idx][2]) # (batch_size, 1)

            review_input_list.append(self.train_data[data_idx][3]) #(batch_size, seq_length)
            review_output_list.append(self.train_data[data_idx][4]) #(batch_size, seq_length)

            
            user, item = self.train_data[data_idx][0], self.train_data[data_idx][1]
            #import pdb; pdb.set_trace()
            user_doc_list.append(self.user_doc_dict[user])
            item_doc_list.append(self.item_doc_dict[item])
            
        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(review_output_list)).cuda(), \
        torch.LongTensor(user_doc_list).cuda(), \
        torch.LongTensor(item_doc_list).cuda()

class TestData():
    def __init__(self, train_data, user_doc_dict, item_doc_dict):
        self.train_data = train_data
        self.length = len(train_data.keys())

        self.user_doc_dict = user_doc_dict
        self.item_doc_dict = item_doc_dict
    
    def get_batch(self, batch_idx_list):
        user_list, item_list = [], []
        review_input_list, real_review_list = [], []

        user_doc_list, item_doc_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0]) # (batch_size, 1)
            item_list.append(self.train_data[data_idx][1]) # (batch_size, 1)

            review_input_list.append(self.train_data[data_idx][3]) #(batch_size, seq_length)
            real_review_list.append(self.train_data[data_idx][5]) #(batch_size, seq_length) real review without PAD

            user, item = self.train_data[data_idx][0], self.train_data[data_idx][1]

            user_doc_list.append(self.user_doc_dict[user])
            item_doc_list.append(self.item_doc_dict[item])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(real_review_list).cuda(), \
        torch.LongTensor(user_doc_list).cuda(), \
        torch.LongTensor(item_doc_list).cuda()