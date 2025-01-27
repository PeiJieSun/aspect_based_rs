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
    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    ############################## CREATE MODEL ##############################
    from pmf import pmf
    model = pmf()

    model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_sports/train_amazon_sports_pmf_id_adam.mod'))
    model.cuda()
    #optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)

    #import adabound
    #optimizer = adabound.AdaBound(model.parameters(), lr=conf.learning_rate, final_lr=0.1, weight_decay=conf.weight_decay)

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_pmf_id_adam.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_pmf_id_adam.log' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_pmf_id_adam.mod' % (conf.out_path, conf.data_name)

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

        train_loss, train_prediction = [], []
        for batch_idx_list in train_batch_sampler:
            user_list, item_list, rating_list = train_dataset.get_batch(batch_idx_list)
            prediction, mse_loss, rmse_loss = model(user_list, item_list, rating_list)
            #import pdb; pdb.set_trace()
            train_loss.extend(tensorToScalar(rmse_loss))
            train_prediction.extend(tensorToScalar(prediction))
            #model.zero_grad(); mse_loss.backward(); optimizer.step()
        t1 = time()

        #import pdb; pdb.set_trace()
        # evaluate the performance of the model with following xxx 
        model.eval()
        
        val_loss, val_prediction = [], []
        for batch_idx_list in val_batch_sampler:
            user_list, item_list, rating_list = train_dataset.get_batch(batch_idx_list)
            prediction, _, rmse_loss = model(user_list, item_list, rating_list)    
            val_loss.extend(tensorToScalar(rmse_loss))
            val_prediction.extend(tensorToScalar(prediction))
        t2 = time()

        test_loss, test_prediction = [], []
        for batch_idx_list in test_batch_sampler:
            user_list, item_list, rating_list = train_dataset.get_batch(batch_idx_list)
            prediction, _, rmse_loss = model(user_list, item_list, rating_list)    
            test_loss.extend(tensorToScalar(rmse_loss))
            test_prediction.extend(tensorToScalar(prediction))
        t3 = time()

        #import pdb; pdb.set_trace()

        train_loss, val_loss, test_loss = np.mean(train_loss), np.mean(val_loss), np.mean(test_loss)

        if epoch == 1:
            min_loss = val_loss
        if val_loss <= min_loss:
            torch.save(model.state_dict(), train_model_path)
        min_loss = min(val_loss, min_loss)

        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_loss, val_loss, test_loss))

        import pdb; pdb.set_trace()