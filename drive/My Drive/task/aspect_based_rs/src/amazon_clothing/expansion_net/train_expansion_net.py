import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time, strftime
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_expansion_net as data_utils
import config_expansion_net as conf

from Logging import Logging

def now():
    return str(strftime('%Y-%m-%d %H:%M:%S'))

def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == '__main__':
    ############################## CREATE MODEL ##############################
    from expansion_net import expansion_net
    model = expansion_net()

    # load word embedding from pretrained word2vec model
    model_params = model.state_dict()
    
    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(3):
        model_params['word_embedding.weight'][idx] = torch.zeros(conf.word_dimension)
    for idx in range(3, conf.vocab_sz):
        model_params['word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-3]])

    model.load_state_dict(model_params)
    

    #model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_expansion_net_id_18.mod'))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_expansion_net_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_expansion_net_id_23.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_expansion_net_id_23.mod' % (conf.out_path, conf.data_name)

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

        train_dataset.construct_aspect_voab()
        aspect_count = train_dataset.count_aspect_words()
        log.record('The number of the words which are aspect words is:%d' % aspect_count)

        review_aspect, review_aspect_bool = train_dataset.construct_aspect_voab()

        model.train()

        train_loss = []
        for batch_idx_list in val_batch_sampler:
            user, item, label, review_input, review_output = train_dataset.get_batch(batch_idx_list)
            generation_loss = model(user, item, label, review_input, \
                review_output, review_aspect, review_aspect_bool)
            train_loss.extend([generation_loss.item()]*len(batch_idx_list))
            model.zero_grad(); generation_loss.backward(); optimizer.step()
        t2 = time()

        # evaluate the performance of the model with following xxx 
        model.eval()
        
        val_loss = []
        for batch_idx_list in val_batch_sampler:
            user, item, label, review_input, review_output = val_dataset.get_batch(batch_idx_list)
            generation_loss = model(user, item, label, review_input, \
                review_output, review_aspect, review_aspect_bool)
            val_loss.extend([generation_loss.item()]*len(batch_idx_list))
        t2 = time()

        test_loss = []
        for batch_idx_list in test_batch_sampler:
            user, item, label, review_input, review_output = test_dataset.get_batch(batch_idx_list)
            generation_loss = model(user, item, label, review_input, \
                review_output, review_aspect, review_aspect_bool)
            test_loss.extend([generation_loss.item()]*len(batch_idx_list))
        t3 = time()
        
        train_loss, val_loss, test_loss = np.mean(train_loss), np.mean(val_loss), np.mean(test_loss)

        if epoch == 1:
            min_loss = val_loss
        if val_loss <= min_loss:
            torch.save(model.state_dict(), train_model_path)
            best_epoch = epoch
        min_loss = min(min_loss, val_loss)
        
        log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('Train:{:.4f}, Val:{:.4f}, Test:{:.4f}'.format(train_loss, val_loss, test_loss))

        #import sys; sys.exit(0)
    print("----"*20)
    print(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    print("----"*20)