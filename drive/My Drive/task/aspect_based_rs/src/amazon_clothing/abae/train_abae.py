import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_abae as data_utils
import config_abae as conf

from Logging import Logging

def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == '__main__':
    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data, train_review_embedding, \
        val_review_embedding, test_review_embedding = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    ############################## CREATE MODEL ##############################
    from abae import abae
    model = abae()
    
    '''
    model_params = model.state_dict()
    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(3):
        model_params['word_embedding.weight'][idx] = torch.zeros(conf.word_dimension)
    for idx in range(3, conf.vocab_sz):
        model_params['word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-3]])

    k_means_weight = np.load('%s/%s.k_means.npy' % (conf.target_path, conf.data_name))
    model_params['transform_T.weight'] = torch.FloatTensor(k_means_weight.transpose()) # (aspect_dimesion, word_dimension)
    
    model.load_state_dict(model_params)
    '''

    model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_abae_id_01.mod'))
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    #import adabound
    #optimizer = adabound.AdaBound(model.parameters(), lr=conf.learning_rate, final_lr=0.1)

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_abae_id_x.py' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_abae_id_02.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_abae_id_02.mod' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData2(train_data, train_review_embedding)
    val_dataset = data_utils.TrainData2(val_data, val_review_embedding)
    test_dataset = data_utils.TrainData2(test_data, test_review_embedding)

    train_batch_sampler = data.BatchSampler(data.RandomSampler(range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(data.RandomSampler(range(val_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    test_batch_sampler = data.BatchSampler(data.RandomSampler(range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    min_loss = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        train_loss = []
        for batch_idx_list in train_batch_sampler:
            _, _, _, review_input, neg_review = train_dataset.get_batch(batch_idx_list)
            out_loss, obj = model(review_input, neg_review)
            train_loss.extend(tensorToScalar(out_loss))
            #model.zero_grad(); obj.backward(); optimizer.step()
        t1 = time()

        # evaluate the performance of the model with following code
        model.eval()
        
        val_loss = []
        for batch_idx_list in val_batch_sampler:
            _, _, _, review_input, neg_review = val_dataset.get_batch(batch_idx_list)
            out_loss, _ = model(review_input, neg_review)    
            val_loss.extend(tensorToScalar(out_loss))
        t2 = time()

        test_loss = []
        for batch_idx_list in test_batch_sampler:
            _, _, _, review_input, neg_review = test_dataset.get_batch(batch_idx_list)
            out_loss, _ = model(review_input, neg_review)    
            test_loss.extend(tensorToScalar(out_loss))
        t3 = time()

        train_loss, val_loss, test_loss = np.mean(train_loss), np.mean(val_loss), np.mean(test_loss)

        if epoch == 1:
            min_loss = val_loss
        if val_loss < min_loss:
            torch.save(model.state_dict(), train_model_path)
        min_loss = min(val_loss, min_loss)

        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_loss, val_loss, test_loss))

        import sys; sys.exit(0)
        #import pdb; pdb.set_trace()