import torch

import numpy as np 
from collections import defaultdict

import torch.utils.data as data
import config_abae_rs as conf

from copy import deepcopy

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

PAD = 0

def load_all():
    user_seq_dict = defaultdict(list)
    item_seq_dict = defaultdict(list)

    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, abae_review = \
            line['user'], line['item'], line['rating'], line['abae_review']
        train_data[idx] = [user, item, rating, abae_review]

        for sent in abae_review:
            sent = sent[:conf.seq_len]
            sent.extend([PAD]*(conf.seq_len - len(sent)))
            user_seq_dict[user].append(sent)
            item_seq_dict[item].append(sent)

    val_data = {}
    f = open(val_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, abae_review = \
            line['user'], line['item'], line['rating'], line['abae_review']
        val_data[idx] = [user, item, rating, abae_review]
    
    test_data = {}
    f = open(test_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, abae_review = \
            line['user'], line['item'], line['rating'], line['abae_review']
        test_data[idx] = [user, item, rating, abae_review]

    for user in user_seq_dict:
        if len(user_seq_dict[user]) < conf.user_seq_num:
            user_seq_dict[user].extend([[PAD]*conf.seq_len]*(conf.user_seq_num-len(user_seq_dict[user])))
        else:
            user_seq_dict[user] = user_seq_dict[user][:conf.user_seq_num]

    for item in item_seq_dict:
        if len(item_seq_dict[item]) < conf.item_seq_num:
            item_seq_dict[item].extend([[PAD]*conf.seq_len]*(conf.item_seq_num-len(item_seq_dict[item])))
        else:
            item_seq_dict[item] = item_seq_dict[item][:conf.item_seq_num]

    '''
    user_seq_num_list = []
    item_seq_num_list = []
    for user in user_seq_dict:
        user_seq_num_list.append(len(user_seq_dict[user]))
    for item in item_seq_dict:
        item_seq_num_list.append(len(item_seq_dict[item]))

    user_seq_num_list.sort()
    item_seq_num_list.sort()

    print('user:%d' % user_seq_num_list[int(0.85*len(user_seq_num_list))])
    print('item:%d' % item_seq_num_list[int(0.85*len(item_seq_num_list))])
    '''

    return train_data, val_data, test_data, user_seq_dict, item_seq_dict

class TrainData():
    def __init__(self, train_data, user_seq_dict, item_seq_dict):
        self.train_data = train_data
        self.user_seq_dict = user_seq_dict
        self.item_seq_dict = item_seq_dict

        self.length = len(train_data.keys())

    def get_batch(self, batch_idx_list):
        user_list, item_list, label_list = [], [], []
        user_pos_sent, user_neg_sent = [], []
        item_pos_sent, item_neg_sent = [], []
        for idx in batch_idx_list:
            user, item, label = self.train_data[idx][0], self.train_data[idx][1], self.train_data[idx][2]

            user_list.append(user)
            item_list.append(item)
            label_list.append(label)
            
            try: 
                user_pos_sent.extend(self.user_seq_dict[user])
                for _ in self.user_seq_dict[user]:
                    for _ in range(conf.num_neg_sent):
                        j = np.random.randint(conf.num_users-1)
                        while j == user or j not in self.user_seq_dict:
                            j = np.random.randint(self.length-1)
                        xx = np.random.randint(conf.user_seq_num-1)
                        user_neg_sent.append(self.user_seq_dict[j][xx])

                item_pos_sent.extend(self.item_seq_dict[item])
                for _ in self.item_seq_dict[item]:
                    for _ in range(conf.num_neg_sent):
                        j = np.random.randint(conf.num_items-1)
                        while j == item or j not in self.item_seq_dict:
                            j = np.random.randint(self.length-1)
                        xx = np.random.randint(conf.item_seq_num-1)
                        item_neg_sent.append(self.item_seq_dict[j][xx])
            except:
                import pdb; pdb.set_trace()

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(label_list).cuda(), \
        torch.LongTensor(user_pos_sent).cuda(), \
        torch.LongTensor(user_neg_sent).cuda(), \
        torch.LongTensor(item_pos_sent).cuda(), \
        torch.LongTensor(item_neg_sent).cuda()