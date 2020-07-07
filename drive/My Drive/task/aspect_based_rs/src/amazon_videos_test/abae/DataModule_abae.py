import torch

import numpy as np 
from collections import defaultdict

import torch.utils.data as data
import config_abae as conf

from copy import deepcopy

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

PAD = 0

def load_all():
    train_data = {}
    f = open(train_data_path)

    sent_id = 0
    for line in f:
        line = eval(line)
        abae_review = line['abae_review']
        for sent in abae_review:
            if len(sent) > conf.seq_len:
                sent = sent[:conf.seq_len]
            else:
                sent.extend([PAD]*(conf.seq_len-len(sent)))
            train_data[sent_id] = sent
            sent_id += 1

    val_data = {}
    f = open(val_data_path)

    sent_id = 0
    for line in f:
        line = eval(line)
        abae_review = line['abae_review']
        for sent in abae_review:
            if len(sent) > conf.seq_len:
                sent = sent[:conf.seq_len]
            else:
                sent.extend([PAD]*(conf.seq_len-len(sent)))
            val_data[sent_id] = sent
            sent_id += 1
    
    test_data = {}
    f = open(test_data_path)

    sent_id = 0
    for line in f:
        line = eval(line)
        abae_review = line['abae_review']
        for sent in abae_review:
            if len(sent) > conf.seq_len:
                sent = sent[:conf.seq_len]
            else:
                sent.extend([PAD]*(conf.seq_len-len(sent)))
            test_data[sent_id] = sent
            sent_id += 1

    return train_data, val_data, test_data

class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

    def get_batch(self, batch_idx_list):
        pos_sent, neg_sent = [], []
        for idx in batch_idx_list:
            pos_sent.append(self.train_data[idx])
        
            for _ in range(conf.num_neg_sent):
                j = np.random.randint(self.length-1)
                while j == idx:
                    j = np.random.randint(self.length-1)
                neg_sent.append(self.train_data[j])

        return torch.LongTensor(pos_sent).cuda(), \
        torch.LongTensor(neg_sent).cuda()