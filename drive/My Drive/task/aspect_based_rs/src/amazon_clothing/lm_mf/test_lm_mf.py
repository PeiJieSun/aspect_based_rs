import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from collections import defaultdict

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_lm_mf as data_utils
import config_lm_mf as conf

from Logging import Logging

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

    model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_clothing/train_amazon_clothing_lm_id_x.mod'))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    test_dataset = data_utils.TrainData(test_data)
    test_batch_sampler = data.BatchSampler(data.RandomSampler(range(test_dataset.length)), batch_size=1, drop_last=False)

    word_dict = constructDict()

    test_loss = []
    for batch_idx_list in test_batch_sampler:
        user_list, item_list, _, review_input_list, review_output_list = test_dataset.get_batch(batch_idx_list)
        sample_idx_list = model.sampleTextByBeamSearch(user_list, item_list)
        convertWord(sample_idx_list, word_dict)
        import pdb; pdb.set_trace()