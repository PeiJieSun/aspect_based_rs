import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np
from collections import defaultdict

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_language_model as data_utils
import config_language_model as conf

from Logging import Logging


from bleu import *
from rouge import *

PAD = 0; SOS = 1; EOS = 2

generation_dict = dict()

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

def evaluate(batch_idx, hyp, ref):
    #random_list = np.random.choice(len(hyp), 1000, replace=False)
    '''
    random_list = range(256)
    new_hyp, new_ref = [], []
    for index in random_list:
        new_hyp.append(hyp[index])
        new_ref.append(ref[index])
    '''
    #compute_bleu(new_hyp, [new_ref])
    #rouge(new_hyp, new_ref)
    print('batch_idx:%d' % batch_idx)
    generation_dict[batch_idx] = hyp
    compute_bleu([hyp], [[ref]])

def generate(batch_idx_list, user, item, review_output):
    current_sequence = defaultdict(list)

    # initialize the dictionary which stores the generated words token idx
    for batch_idx in batch_idx_list:
        current_sequence[batch_idx] = [SOS]

    # first iteration: prepare the input text data
    current_word_input = torch.LongTensor([[SOS] * len(batch_idx_list)]).view(1, -1).cuda() # (1, batch_size)
    user_embed = model.user_embedding(user)
    item_embed = model.item_embedding(item)

    hidden_state = \
        model.linear_1(torch.cat([user_embed, item_embed], 1)).view(1, -1, conf.hidden_size) #(1, batch_size, hidden_size)
        
    for _ in range(29):
        word_probit, hidden_state = model.generate(hidden_state, current_word_input) # review_output_embedding: (batch_size*beam_size, word_dimension)        
        indices = torch.argmax(word_probit, 1) # (batch_size, 1)
        current_word_input = indices.view(1, -1) # (1, batch_size)
        
        #import pdb; pdb.set_trace()
        for out_idx, batch_idx in enumerate(batch_idx_list):
            current_sequence[batch_idx].append(indices[out_idx].item())
    
    word_dict = constructDict()

    hyp = []
    for out_idx, batch_idx in enumerate(batch_idx_list):
        convertWord(current_sequence[out_idx], word_dict)
        #tmp_hyp = current_sequence[out_idx][0]
        #hyp.append(tmp_hyp)
        #evaluate(batch_idx, tmp_hyp, review_output[out_idx])

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    ############################## LOAD MODEL ##############################
    from language_model import language_model
    model = language_model()

    model.load_state_dict(
        torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_clothing/train_amazon_clothing_language_model_id_0X.mod'))

    model.cuda()

    ########################### TEST STAGE #####################################
    test_dataset = data_utils.TrainData(test_data)

    test_batch_sampler = data.BatchSampler(data.SequentialSampler(range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    t0 = time()
    model.eval()

    for batch_idx_list in test_batch_sampler:
        user, item, _, review_input, review_output = test_dataset.get_batch(batch_idx_list)
        generate(batch_idx_list, user, item, tensorToScalar(review_output.transpose(0, 1)).tolist())
    