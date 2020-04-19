import torch
import torch.nn.functional as F

import numpy as np 
from collections import defaultdict
from gensim.models import Word2Vec
import config_expansion_net as conf

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
    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review = line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review)
        train_data[idx] = [user, item, rating, review_in, review_out]

    val_data = {}
    f = open(val_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review = line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review)
        val_data[idx] = [user, item, rating, review_in, review_out]
    
    test_data = {}
    f = open(test_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review = line['user'], line['item'], line['rating'], line['g_review']
        review_in, review_out = generate_review(g_review)
        test_data[idx] = [user, item, rating, review_in, review_out]
        
    return train_data, val_data, test_data
    
class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

        self.construct_aspect_voab()

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list = [], []
        review_aspect_bool_list, review_aspect_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0]) # (batch_size, 1)
            item_list.append(self.train_data[data_idx][1]) # (batch_size, 1)
            rating_list.append(self.train_data[data_idx][2]) # (batch_size, 1)

            review_input_list.append(self.train_data[data_idx][3]) #(batch_size, seq_length)
            review_output_list.append(self.train_data[data_idx][4]) #(batch_size, seq_length)

            # construct review_aspect_list, review_aspect_bool_list
            tmp_review_aspect_list, tmp_review_aspect_bool_list = [], []
            for word_id in self.train_data[data_idx][4]:
                if word_id in self.aspect_vocab:
                    tmp_review_aspect_bool_list.append([1])
                    tmp_review_aspect_list.append(self.aspect_vocab[word_id])
                else:
                    tmp_review_aspect_bool_list.append([0])
                    tmp_review_aspect_list.append(0)
            review_aspect_bool_list.append(tmp_review_aspect_bool_list) # (batch_size, seq_length)
            review_aspect_list.append(tmp_review_aspect_list) # (batch_size, seq_length)

        review_aspect_list = np.transpose(review_aspect_list) # (seq_length, batch_size)
        review_aspect_bool_list = np.transpose(review_aspect_bool_list) # (seq_length, batch_size)

        review_aspect_list = review_aspect_list.reshape(-1, 1) #(seq_length*batch_size, 1)
        review_aspect_bool_list = review_aspect_bool_list.reshape(-1, 1) # (seq_length*batch_size, 1)

        
        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(review_output_list)).cuda(), \
        torch.LongTensor(review_aspect_list).cuda(), \
        torch.LongTensor(review_aspect_bool_list).cuda()
        '''
        return torch.LongTensor(user_list), \
        torch.LongTensor(item_list), \
        torch.FloatTensor(rating_list), \
        torch.LongTensor(np.transpose(review_input_list)), \
        torch.LongTensor(np.transpose(review_output_list)), \
        torch.LongTensor(review_aspect_list), \
        torch.LongTensor(review_aspect_bool_list)
        '''
    def construct_aspect_voab(self):
        aspect_vocab = {}

        aspect_params = torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_aspect_rating_3_id_59.mod')
        c = aspect_params['transform_T.weight'].transpose(0, 1) # (aspect_dimesion, word_dimension)
        x = aspect_params['word_embedding.weight'] # (num_words, word_dimension)

        x_i = F.normalize(x[:, None, :], p=2, dim=2) # (num_words, 1, word_dimension)
        c_j = F.normalize(c[None, :, :], p=2, dim=2) # (1, aspect_dimesion, word_dimension)
        
        D_ij = torch.transpose((x_i * c_j).sum(-1), 0, 1) # (aspect_dimesion, num_words)

        K = 100
        _, indices = torch.topk(D_ij, K) # (aspect_dimesion, K)

        word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))

        for idx, value in enumerate(indices):
            for word_id in value:
                aspect_vocab[word_id] = idx
        self.aspect_vocab = aspect_vocab

    def count_aspect_words(self):
        aspect_count = 0
        for value in self.train_data:
            review = value[4]
            for word_id in review:
                if word_id in self.aspect_vocab:
                    aspect_count += 1
        return aspect_count