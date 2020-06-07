import os, sys, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np

from collections import defaultdict

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_lm as data_utils
import config_lm as conf

from bleu import *
from rouge import rouge

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

def constructDict():
    word_dict = np.load('/content/drive/My Drive/task/aspect_based_rs/data/amazon_musical_test/amazon_musical_test.vocab_decoder.npy', allow_pickle=True).item()

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

    model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_musical_test/train_amazon_musical_test_lm_id_X8_40.mod'))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    import pdb; pdb.set_trace()
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

    model.eval()

    t0 = time()
    count = 0
    bleu_score = []
    bleu_list_1, bleu_list_2, bleu_list_3, bleu_list_4 = [], [], [], []
    rouge_1_list, rouge_2_list, rouge_L_list = [], [], []
    out_probit = []; target = []
    for batch_idx_list in test_batch_sampler:
        user_list, item_list, review_input_list, _, real_review_list = test_dataset.get_batch(batch_idx_list)
        sample_idx_list, _ = model._sample_text_by_top_one(user_list, item_list, review_input_list)
        #convertWord(sample_idx_list, word_dict)
        ref = tensorToScalar(real_review_list).tolist()
        try:
            bleu_score = compute_bleu([sample_idx_list], [ref])
            bleu_list_1.append(bleu_score[1])
            bleu_list_2.append(bleu_score[2])
            bleu_list_3.append(bleu_score[3])
            bleu_list_4.append(bleu_score[4])

            rouge_score = rouge([sample_idx_list], ref)
            rouge_1_list.append(rouge_score[0])
            rouge_2_list.append(rouge_score[1])
            rouge_L_list.append(rouge_score[2])
        except:
            pass
        count += 1
        if count % 50 == 0:
            import sys; sys.exit(0)
            t1 = time()
            print('Generating %d lines, test samples cost:%.4fs' % (count, (t1-t0)))

    print('bleu_1:%.4f' % np.mean(bleu_list_1))
    print('bleu_2:%.4f' % np.mean(bleu_list_2))
    print('bleu_3:%.4f' % np.mean(bleu_list_3))
    print('bleu_4:%.4f' % np.mean(bleu_list_4))
    print('rouge_1_f:%.4f' % np.mean(rouge_1_list))
    print('rouge_2_f:%.4f' % np.mean(rouge_2_list))
    print('rouge_L_f:%.4f' % np.mean(rouge_L_list))
    t1 = time()
    print('Generating all test samples cost:%.4fs' % (t1-t0))