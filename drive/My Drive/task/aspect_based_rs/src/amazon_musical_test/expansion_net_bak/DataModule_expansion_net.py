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

def generate_review(review, summary):
    '''
    review_in = [SOS]
    review_in.extend(review)
    review_out = review
    review_out.append(EOS)
    '''
    review_in = review[:conf.rev_len-1]
    review_out = review[1:conf.rev_len]
    review_in.extend([PAD]*(conf.rev_len-len(review_in)))
    review_out.extend([PAD]*(conf.rev_len-len(review_out)))

    summary = summary[:conf.sum_len]
    summary.extend([PAD]*(conf.sum_len-len(summary)))
    return review_in, review_out, summary

def load_all():
    max_user, max_item = 0, 0

    train_data = {}
    f = open(train_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review, summary = \
            line['user'], line['item'], line['rating'], line['g_review'], line['summary']
        review_in, review_out, summary = generate_review(g_review, summary)

        g_review = g_review[:conf.rev_len]
        g_review.extend([PAD]*(conf.rev_len-len(g_review)))
        train_data[idx] = [user, item, rating, review_in, review_out, summary, g_review]

        max_user = max(max_user, user)
        max_item = max(max_item, item)

    val_data = {}
    f = open(val_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review, summary = \
            line['user'], line['item'], line['rating'], line['g_review'], line['summary']
        review_in, review_out, summary = generate_review(g_review, summary)

        g_review = g_review[:conf.rev_len]
        g_review.extend([PAD]*(conf.rev_len-len(g_review)))
        val_data[idx] = [user, item, rating, review_in, review_out, summary, g_review]
    
    test_data = {}
    f = open(test_data_path)
    for idx, line in enumerate(f):
        line = eval(line)
        user, item, rating, g_review, summary = \
            line['user'], line['item'], line['rating'], line['g_review'], line['summary']
        review_in, review_out, summary = generate_review(g_review, summary)

        g_review = g_review[:conf.rev_len]
        g_review.extend([PAD]*(conf.rev_len-len(g_review)))
        test_data[idx] = [user, item, rating, review_in, review_out, summary, g_review]
    
    print('max_user:%d, max_item:%d' % (max_user, max_item))
    return train_data, val_data, test_data
    
class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

        #self.construct_aspect_voab()

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list = [], []
        summary_list = []
        review_aspect_bool_list, review_aspect_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.train_data[data_idx][0]) # (batch_size, 1)
            item_list.append(self.train_data[data_idx][1]) # (batch_size, 1)
            rating_list.append(self.train_data[data_idx][2]) # (batch_size, 1)
            
            review_input_list.append(self.train_data[data_idx][3]) #(batch_size, seq_length)
            review_output_list.append(self.train_data[data_idx][4]) #(batch_size, seq_length)

            summary_list.append(self.train_data[data_idx][5]) # (batch_size, sum_len)

        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.FloatTensor(rating_list).cuda(), \
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(review_output_list)).cuda(), \
        torch.LongTensor(np.transpose(summary_list)).cuda()


    def construct_aspect_voab(self):
        aspect_vocab = {}

        aspect_params = torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_musical_test/train_amazon_musical_test_abae_id_X.mod')
        c = aspect_params['transform_T.weight'].transpose(0, 1) # (aspect_dimesion, word_dimension)
        x = aspect_params['word_embedding.weight'] # (num_words, word_dimension)

        x_i = F.normalize(x[:, None, :], p=2, dim=2) # (num_words, 1, word_dimension)
        c_j = F.normalize(c[None, :, :], p=2, dim=2) # (1, aspect_dimesion, word_dimension)
        
        D_ij = torch.transpose((x_i * c_j).sum(-1), 0, 1) # (aspect_dimesion, num_words)

        K = 100
        _, indices = torch.topk(D_ij, K) # (aspect_dimesion, K)

        abae_decoder_dict = np.load('/content/drive/My Drive/task/aspect_based_rs/data/amazon_musical_test/amazon_musical_test.abae_vocab_decoder.npy', allow_pickle=True).item()
        g_encoder_dict = np.load('/content/drive/My Drive/task/aspect_based_rs/data/amazon_musical_test/amazon_musical_test.g_vocab.npy', allow_pickle=True).item()
        g_decoder_dict = np.load('/content/drive/My Drive/task/aspect_based_rs/data/amazon_musical_test/amazon_musical_test.g_vocab_decoder.npy', allow_pickle=True).item()
        
        count = 0
        aspect_word = set()
        for idx, word_idx_list in enumerate(indices):
            #aspect_word_list = 'aspect_%d: ' % (idx+1)
            #veryify_word_list = 'veryify aspect_%d: ' % (idx+1)
            for word_idx in word_idx_list:
                #aspect_word_list += '%s, ' % abae_decoder_dict[word_idx.item()]
                
                abae_word = abae_decoder_dict[word_idx.item()]
                if abae_word in g_encoder_dict:
                    count += 1
                    g_word_id = g_encoder_dict[abae_word]
                aspect_vocab[g_word_id] = idx
                aspect_word.add(g_word_id)

                #veryify_word_list += '%s, ' % g_decoder_dict[g_word_id]

        #import pdb; pdb.set_trace()
        self.aspect_vocab = aspect_vocab

        review_aspect, review_aspect_mask = [], []
        for word_idx in range(conf.vocab_sz):
            if word_idx in aspect_vocab:
                review_aspect.append(aspect_vocab[word_idx])
                review_aspect_mask.append(1)
            else:
                review_aspect.append(0)
                review_aspect_mask.append(0)

        return torch.LongTensor(review_aspect).cuda(), \
            torch.LongTensor(review_aspect_mask).view(1, -1).cuda()

    def count_aspect_words(self):
        aspect_count = 0
        for key, value in self.train_data.items():
            review = value[4]
            for word_id in review:
                if word_id in self.aspect_vocab:
                    aspect_count += 1
        return aspect_count


class TestData():
    def __init__(self, test_data):
        self.test_data = test_data
        self.length = len(test_data.keys())

        #self.construct_aspect_voab()

    def get_batch(self, batch_idx_list):
        user_list, item_list, rating_list = [], [], []
        review_input_list, review_output_list = [], []
        real_review_list = []
        summary_list = []
        review_aspect_bool_list, review_aspect_list = [], []

        for data_idx in batch_idx_list:
            user_list.append(self.test_data[data_idx][0]) # (batch_size, 1)
            item_list.append(self.test_data[data_idx][1]) # (batch_size, 1)
            rating_list.append(self.test_data[data_idx][2]) # (batch_size, 1)
            
            review_input_list.append(self.test_data[data_idx][3]) #(batch_size, seq_length)
            summary_list.append(self.test_data[data_idx][5]) # (batch_size, sum_len)

            real_review_list.append(self.test_data[data_idx][6]) #(batch_size, seq_length)

            #import pdb; pdb.set_trace()
        return torch.LongTensor(user_list).cuda(), \
        torch.LongTensor(item_list).cuda(), \
        torch.LongTensor(np.transpose(review_input_list)).cuda(), \
        torch.LongTensor(np.transpose(summary_list)).cuda(), \
        torch.LongTensor(real_review_list).cuda()