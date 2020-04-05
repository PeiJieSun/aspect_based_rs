import torch
import torch.nn.functional as F

import numpy as np 
from collections import defaultdict
from collections import OrderedDict

import torch.utils.data as data
import config_generation as conf

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

train_data_path = '%s/%s.train.data' % (conf.target_path, conf.data_name)
val_data_path = '%s/%s.val.data' % (conf.target_path, conf.data_name)
test_data_path = '%s/%s.test.data' % (conf.target_path, conf.data_name)

def generate_review(review):
    review_in = [SOS]
    review_in.extend(review)
    review_out = review
    review_out.append(EOS)
    return review_in, review_out

def load_all():
    train_user_historical_review_dict, train_item_historical_review_dict = defaultdict(list), defaultdict(list)

    train_data = {}
    f = open(train_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating, review, g_review = \
            line['idx'], line['user'], line['item'], line['rating'], line['review'], line['g_review']
        review_in, review_out = generate_review(g_review)
        train_data[idx] = [user, item, rating, review, review_in, review_out]
        if len(train_user_historical_review_dict[user]) < conf.u_max_r:
            train_user_historical_review_dict[user].append(idx)
        if len(train_item_historical_review_dict[item]) < conf.i_max_r:
            train_item_historical_review_dict[item].append(idx)

    val_data = {}
    f = open(val_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
        val_data[idx] = [user, item, rating]
    
    test_data = {}
    f = open(test_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
        test_data[idx] = [user, item, rating]
        
    return train_data, val_data, test_data, train_user_historical_review_dict, train_item_historical_review_dict
    
class TrainData():
    def __init__(self, train_data, 
        user_historical_review_dict, 
        item_historical_review_dict, model):
        self.model = model
        self.train_data = train_data
        self.length = len(train_data.keys())
        self.user_historical_review_dict = user_historical_review_dict
        self.item_historical_review_dict = item_historical_review_dict

        self.construct_aspect_voab()

    def get_batch(self, batch_idx_list):
        old_user_histor_idx, old_item_histor_idx = OrderedDict(), OrderedDict()
        review_idx_set = set()
        
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list = [], []
        review_aspect_bool_list, review_aspect_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0])
            item_list.append(self.train_data[data_idx][1])
            rating_list.append(self.train_data[data_idx][2])

            review_input_list.append(self.train_data[data_idx][4])
            review_output_list.append(self.train_data[data_idx][5])

            tmp_review_aspect_list, tmp_review_aspect_bool_list = [], []
            for word_id in self.train_data[data_idx][5]:
                if word_id in self.aspect_vocab:
                    tmp_review_aspect_bool_list.append([1])
                    tmp_review_aspect_list.append(self.aspect_vocab[word])
                else:
                    tmp_review_aspect_bool_list.append([0])
                    tmp_review_aspect_list.append([0])
            review_aspect_bool_list.append(tmp_review_aspect_bool_list)
            review_aspect_list.append(tmp_review_aspect_list)
            
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
        historical_review_list = []
        for review_idx in review_idx_list:
            historical_review_list.append(self.train_data[review_idx][3])
            for _ in range(conf.num_negative_reviews):
                j = np.random.randint(self.length-1)
                while j == idx:
                    j = np.random.randint(self.length-1)
                neg_review.append(self.train_data[j][3])

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.array(historical_review_list)).cuda(),\
        torch.LongTensor(np.array(neg_review)).cuda(),\
        torch.LongTensor(user_histor_index).cuda(),\
        torch.FloatTensor(user_histor_value).cuda(),\
        torch.LongTensor(item_histor_index).cuda(),\
        torch.FloatTensor(item_histor_value).cuda(),\
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(review_output_list)).cuda(), \
        torch.LongTensor(review_aspect_bool_list).cuda(), \
        torch.LongTensor(review_aspect_list).cuda()

    def construct_aspect_voab(self):
        aspect_vocab = {}

        c = self.model.transform_T.weight.transpose(0, 1) # (aspect_dimesion, word_dimension)
        x = self.model.word_embedding.weight # (num_words, word_dimension)

        x_i = F.normalize(x[:, None, :], p=2, dim=2) # (num_words, 1, word_dimension)
        c_j = F.normalize(c[None, :, :], p=2, dim=2) # (1, aspect_dimesion, word_dimension)
        
        D_ij = torch.transpose((x_i * c_j).sum(-1), 0, 1) # (aspect_dimesion, num_words)

        K = 100
        _, indices = torch.topk(D_ij, K) # (aspect_dimesion, K)

        for idx, value in enumerate(indices):
            for word_id in value:
                aspect_vocab[value] = idx                
        self.aspect_vocab = aspect_vocab


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