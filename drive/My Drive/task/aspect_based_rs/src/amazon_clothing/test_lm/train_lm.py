import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_lm as data_utils
import config_lm as conf

from Logging import Logging

def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == '__main__':
    ############################## CREATE MODEL ##############################
    from lm import lm
    model = lm()

    model_params = model.state_dict()
    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(3):
        model_params['word_embedding.weight'][idx] = torch.zeros(conf.word_dimension)
    for idx in range(3, conf.vocab_sz):
        model_params['word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-3]])
    
    ncf_params = torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_clothing/train_amazon_clothing_ncf_id_X1.mod')
    model_params['encoder.user_embedding.weight'] = ncf_params['mlp_user_embedding.weight']
    model_params['encoder.item_embedding.weight'] = ncf_params['mlp_item_embedding.weight']
    
    model.load_state_dict(model_params)
    
    #model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_clothing/train_amazon_clothing_lm_id_X7.mod'))
    model.cuda()
    
    #model.word_embedding.weight.requires_grad = False
    #model.encoder.user_embedding.weight.requires_grad = False
    #model.encoder.item_embedding.weight.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_lm_id_x.py' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_lm_id_X7.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_lm_id_X7.mod' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data)
    val_dataset = data_utils.TrainData(val_data)
    test_dataset = data_utils.TrainData(test_data)

    train_batch_sampler = data.BatchSampler(data.RandomSampler(range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(data.RandomSampler(range(val_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    test_batch_sampler = data.BatchSampler(data.RandomSampler(range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    min_loss = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        #import pdb; pdb.set_trace()
        train_review_loss = []
        for batch_idx_list in train_batch_sampler:
            user_list, item_list, _, review_input_list, review_output_list = train_dataset.get_batch(batch_idx_list)
            out_loss, obj = model(user_list, item_list, review_input_list, review_output_list)
            train_review_loss.extend(tensorToScalar(out_loss))
            model.zero_grad(); obj.backward(); 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            #import pdb; pdb.set_trace()
        t1 = time()

        # evaluate the performance of the model with following code
        model.eval()
        
        val_review_loss = []
        for batch_idx_list in val_batch_sampler:
            user_list, item_list, _, review_input_list, review_output_list = val_dataset.get_batch(batch_idx_list)
            out_loss, _ = model(user_list, item_list, review_input_list, review_output_list)
            val_review_loss.extend(tensorToScalar(out_loss))
        t2 = time()

        test_review_loss = []
        for batch_idx_list in test_batch_sampler:
            user_list, item_list, _, review_input_list, review_output_list = test_dataset.get_batch(batch_idx_list)
            out_loss, _ = model(user_list, item_list, review_input_list, review_output_list)
            test_review_loss.extend(tensorToScalar(out_loss))
        t3 = time()

        train_review_loss, val_review_loss, test_review_loss = \
            np.mean(train_review_loss), np.mean(val_review_loss), np.mean(test_review_loss)
        
        if epoch == 1:
            min_loss = val_review_loss
        if val_review_loss < min_loss:
            torch.save(model.state_dict(), train_model_path)
        min_loss = min(val_review_loss, min_loss)
        
        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_review_loss, val_review_loss, test_review_loss))

        #import sys; sys.exit()