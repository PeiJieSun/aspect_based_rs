import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_language_model as data_utils
import config_language_model as conf

from Logging import Logging

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == '__main__':
    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    ############################## CREATE MODEL ##############################
    from language_model import language_model
    model = language_model()

    model_params = model.state_dict()
    word_embedding = Word2Vec.load('/content/drive/My Drive/task/aspect_based_rs/data/amazon_movies_tv/amazon_movies_tv.wv.model')
    for idx in range(3, conf.vocab_sz):
        model_params['word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-3]])
    model.load_state_dict(model_params)

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    ########################### FIRST TRAINING #####################################
    log = Logging('/content/drive/My Drive/task/aspect_based_rs/out/amazon_movies_tv/train_amazon_movies_tv_language_model_id_x.log')
    log_detail = Logging('/content/drive/My Drive/task/aspect_based_rs/out/amazon_movies_tv/detail_train_amazon_movies_tv_language_model_id_x.log')
    train_model_path = '/content/drive/My Drive/task/aspect_based_rs/out/amazon_movies_tv/train_amazon_movies_tv_language_model_id_x.mod'

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

        train_loss = []
        for batch_idx_list in train_batch_sampler:
            _, _, _, review_input, review_output = train_dataset.get_batch(batch_idx_list)
            out = model(review_input, review_output)
            train_loss.append(out.loss.item())
            model.zero_grad(); out.loss.backward(); optimizer.step()
        t1 = time()

        # evaluate the performance of the model with following xxx 
        model.eval()
        
        val_loss = []
        for batch_idx_list in val_batch_sampler:
            _, _, _, review_input, review_output = train_dataset.get_batch(batch_idx_list)
            out = model(review_input, review_output)    
            val_loss.append(out.loss.item())
        t2 = time()

        test_loss = []
        for batch_idx_list in test_batch_sampler:
            _, _, _, review_input, review_output = train_dataset.get_batch(batch_idx_list)
            out = model(review_input, review_output)    
            test_loss.append(out.loss.item())
        t3 = time()

        if epoch == 1:
            min_loss = val_loss
        if val_loss <= min_loss:
            torch.save(model.state_dict(), train_model_path)
        min_loss = min(val_loss, min_loss)

        log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(np.sum(train_loss), np.sum(val_loss), np.sum(test_loss)))