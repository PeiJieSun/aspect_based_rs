import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from collections import defaultdict

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_lm as data_utils
import config_lm as conf

from bleu import *

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

def constructDict():
    word_dict = defaultdict()
    # prepare the data x
    wv_model = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))

    for idx, word in enumerate(wv_model.wv.vocab):
        word_dict[idx] = word

    word_dict[0], word_dict[1], word_dict[2] = 'PAD', 'SOS', 'EOS' 

    return word_dict

def convertWord(word_idx_list, word_dict):
    sentence = ''
    for word_idx in word_idx_list:
        sentence += '%s ' % word_dict[word_idx]
    print(sentence)
    
if __name__ == '__main__':
    
    ############################## CREATE MODEL ##############################
    from lm import lm
    model = lm()

    model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_clothing/train_amazon_clothing_lm_id_X1.mod'))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    test_dataset = data_utils.TestData(test_data)
    test_batch_sampler = data.BatchSampler(data.RandomSampler(range(test_dataset.length)), batch_size=1, drop_last=False)

    print('test dataset length:%d' % test_dataset.length)
    word_dict = constructDict()

    test_loss = []
    t0 = time()
    count = 0
    bleu_list_1, bleu_list_2, bleu_list_3, bleu_list_4 = [], [], [], []
    for batch_idx_list in test_batch_sampler:
        user_list, item_list, _, review_output_list = test_dataset.get_batch(batch_idx_list)
        sample_idx_list = model.sampleTextByTopOne(user_list, item_list)
        convertWord(sample_idx_list, word_dict)
        ref = review_output_list
        bleu_score = compute_bleu([sample_idx_list], [[ref]])
        bleu_list_1.append(bleu_score[1])
        bleu_list_2.append(bleu_score[2])
        bleu_list_3.append(bleu_score[3])
        bleu_list_4.append(bleu_score[4])
        #import pdb; pdb.set_trace()
        count += 1
        if count % 50 == 0:
            import sys; sys.exit()
            t1 = time()
            print('Generating %d lines, test samples cost:%.4fs' % (count, (t1-t0)))

    print('compute bleu_1:%.4f' % (np.mean(bleu_list_1)))
    print('compute bleu_2:%.4f' % (np.mean(bleu_list_2)))
    print('compute bleu_3:%.4f' % (np.mean(bleu_list_3)))
    print('compute bleu_4:%.4f' % (np.mean(bleu_list_4)))
    t1 = time()
    print('Generating all test samples cost:%.4fs' % (t1-t0))