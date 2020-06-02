import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_pmf as data_utils
import config_pmf as conf

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
    from pmf import pmf
    model = pmf()
    
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_pmf_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_pmf_id_X.log' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_pmf_id_X.mod' % (conf.out_path, conf.data_name)

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

        train_rating_loss = []
        for batch_idx_list in train_batch_sampler:
            user_list, item_list, rating_list, _, _ = train_dataset.get_batch(batch_idx_list)
            prediction, obj_loss, mse_loss = model(user_list, item_list, rating_list)
            train_rating_loss.extend(tensorToScalar(mse_loss))
            model.zero_grad(); obj_loss.backward(); optimizer.step()
        t1 = time()

        # evaluate the performance of the model with following xxx 
        model.eval()
        
        val_rating_loss = []
        for batch_idx_list in val_batch_sampler:
            user_list, item_list, rating_list, _, _ = val_dataset.get_batch(batch_idx_list)
            prediction, _, mse_loss = model(user_list, item_list, rating_list)    
            val_rating_loss.extend(tensorToScalar(mse_loss))
        t2 = time()

        test_rating_loss = []
        for batch_idx_list in test_batch_sampler:
            user_list, item_list, rating_list, _, _ = test_dataset.get_batch(batch_idx_list)
            prediction, _, mse_loss = model(user_list, item_list, rating_list)    
            test_rating_loss.extend(tensorToScalar(mse_loss))
        t3 = time()

        train_rating_loss, val_rating_loss, test_rating_loss = \
            np.sqrt(np.mean(train_rating_loss)), np.sqrt(np.mean(val_rating_loss)), np.sqrt(np.mean(test_rating_loss))

        if epoch == 1:
            min_loss = val_rating_loss
        if val_rating_loss < min_loss:
            torch.save(model.state_dict(), train_model_path)
        min_loss = min(val_rating_loss, min_loss)

        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_rating_loss, val_rating_loss, test_rating_loss))