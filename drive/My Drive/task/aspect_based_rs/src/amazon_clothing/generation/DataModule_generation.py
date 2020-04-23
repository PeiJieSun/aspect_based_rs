# ABAE + Rating Prediction

import torch
import torch.nn.functional as F

import numpy as np 
from collections import defaultdict
from collections import OrderedDict

import torch.utils.data as data
import config_generation as conf

from gensim.models import Word2Vec

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
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, review, g_review = line['user'], line['item'], line['rating'], line['review'], line['g_review']
        review, _ = generate_review(review)
        review_in, review_out = generate_review(g_review)
        train_data[idx] = [user, item, rating, review, review_in, review_out]

        if len(train_user_historical_review_dict[user]) < conf.u_max_r:
            train_user_historical_review_dict[user].append(idx)
        if len(train_item_historical_review_dict[item]) < conf.i_max_r:
            train_item_historical_review_dict[item].append(idx)

    val_data = {}
    f = open(val_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, review, g_review = line['user'], line['item'], line['rating'], line['review'], line['g_review']
        review, _ = generate_review(review)
        review_in, review_out = generate_review(g_review)
        val_data[idx] = [user, item, rating, review, review_in, review_out]
    
    test_data = {}
    f = open(test_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, review, g_review = line['user'], line['item'], line['rating'], line['review'], line['g_review']
        review, _ = generate_review(review)
        review_in, review_out = generate_review(g_review)
        test_data[idx] = [user, item, rating, review, review_in, review_out]
    
    #import pdb; pdb.set_trace()
    return train_data, val_data, test_data, train_user_historical_review_dict, train_item_historical_review_dict

class TrainData():
    def __init__(self, train_data, 
        user_historical_review_dict, 
        item_historical_review_dict, review_data):
        
        self.review_data = review_data
        self.train_data = train_data
        self.length = len(train_data.keys())
        self.user_historical_review_dict = user_historical_review_dict
        self.item_historical_review_dict = item_historical_review_dict

    def get_batch(self, batch_idx_list):
        old_user_histor_idx, old_item_histor_idx = OrderedDict(), OrderedDict()
        review_idx_set = set()
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0])
            item_list.append(self.train_data[data_idx][1])
            rating_list.append(self.train_data[data_idx][2])

            review_input_list.append(self.train_data[data_idx][4]) #(batch_size, seq_length)
            review_output_list.append(self.train_data[data_idx][5]) #(batch_size, seq_length)

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

        # prepare review positive/negative embedding, and review words input
        neg_review = []
        pos_review = []
        for review_idx in review_idx_list:
            pos_review.append(self.review_data[review_idx][3])
            for _ in range(conf.num_negative_reviews):
                j = np.random.randint(self.length-1)
                while j == review_idx:
                    j = np.random.randint(self.length-1)
                neg_review.append(self.review_data[j][3])
        pos_review.append([PAD]*conf.max_review_len)
        neg_review.append([PAD]*conf.max_review_len)

        user_idx_list, item_idx_list = [], []

        for idx, key in enumerate(old_user_histor_idx):
            old_review_idx_list = old_user_histor_idx[key]
            for review_idx in old_review_idx_list:
                user_idx_list.append(review_mapping[review_idx])
            user_idx_list.extend([len(pos_review)-1]*(conf.u_max_r-len(old_review_idx_list)))
        for idx, key in enumerate(old_item_histor_idx):
            old_review_idx_list = old_item_histor_idx[key]
            for review_idx in old_review_idx_list:
                item_idx_list.append(review_mapping[review_idx])
            item_idx_list.extend([len(pos_review)-1]*(conf.i_max_r-len(old_review_idx_list)))
        
        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.array(pos_review)).cuda(),\
        torch.LongTensor(np.array(neg_review)).cuda(),\
        torch.LongTensor(user_idx_list).cuda(),\
        torch.LongTensor(item_idx_list).cuda(),\
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(review_output_list)).cuda()

    def construct_aspect_voab(self, model):
        aspect_vocab = {}

        #aspect_params = torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_abae_id_01.mod')
        #c = aspect_params['transform_T.weight'].transpose(0, 1) # (aspect_dimesion, word_dimension)
        #x = aspect_params['word_embedding.weight'] # (num_words, word_dimension)

        c = model.transform_T.weight.transpose(0, 1)
        x = model.word_embedding.weight

        x_i = F.normalize(x[:, None, :], p=2, dim=2) # (num_words, 1, word_dimension)
        c_j = F.normalize(c[None, :, :], p=2, dim=2) # (1, aspect_dimesion, word_dimension)
        
        D_ij = torch.transpose((x_i * c_j).sum(-1), 0, 1) # (aspect_dimesion, num_words)

        K = 100
        _, indices = torch.topk(D_ij, K) # (aspect_dimesion, K)

        word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))

        for idx, word_idx_list in enumerate(indices):
            aspect_word_list = 'aspect_%d: ' % (idx+1)
            for word_idx in word_idx_list:
                aspect_word_list += '%s, ' % word_embedding.wv.index2entity[word_idx.item()-3]
                aspect_vocab[word_idx.item()] = idx
        self.aspect_vocab = aspect_vocab

        review_aspect, review_aspect_bool = [], []
        for word_idx in range(conf.vocab_sz):
            if word_idx in aspect_vocab:
                review_aspect.append(aspect_vocab[word_idx])
                review_aspect_bool.append(1)
            else:
                review_aspect.append(0)
                review_aspect_bool.append(0)
        
        return torch.LongTensor(review_aspect).cuda(), torch.LongTensor(review_aspect_bool).cuda().view(1, -1)

    def count_aspect_words(self):
        aspect_count = 0
        for key, value in self.train_data.items():
            review = value[3]
            for word_id in review:
                if word_id in self.aspect_vocab:
                    aspect_count += 1
        return aspect_count