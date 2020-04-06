import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_generation as data_utils
import config_generation as conf

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
    from generation import generation
    model = generation()

    model_params = model.state_dict()
    fm_params = torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_fm_id_2.mod')
    for param in fm_params:
        if param in fm_params:
            model_params[param] = fm_params[param]
    model.load_state_dict(model_params)
    # load word embedding from pretrained word2vec model
    #model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_fm_id_2.mod'))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data, train_user_historical_review_dict,\
        train_item_historical_review_dict = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_generation_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_generation_id_x.log' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_generation_id_x.mod' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data, train_user_historical_review_dict, train_item_historical_review_dict, model)
    val_dataset = data_utils.ValData(val_data)
    test_dataset = data_utils.ValData(test_data)

    train_batch_sampler = data.BatchSampler(data.SequentialSampler(range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(data.SequentialSampler(range(val_dataset.length)), batch_size=5, drop_last=False)
    test_batch_sampler = data.BatchSampler(data.SequentialSampler(range(test_dataset.length)), batch_size=5, drop_last=False)
    #RandomSampler
    
    # Start Training !!!
    min_rating_loss = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        # evaluate the performance of the model with following xxx 
        model.eval()
        
        total_user, total_label = [], []
        val_rating_loss, val_prediction = [], []
        for batch_idx_list in val_batch_sampler:
            user, item, label = val_dataset.get_batch(batch_idx_list)
            total_user.extend(user); total_label.extend(label)
            prediction, rating_loss = model.predict(user, item, label)
            val_prediction.extend(tensorToScalar(prediction)); val_rating_loss.extend(tensorToScalar(rating_loss))
        t2 = time()

        total_user, total_item, total_label = [], [], []
        test_rating_loss, test_prediction = [], []
        for batch_idx_list in test_batch_sampler:
            user, item, label = test_dataset.get_batch(batch_idx_list)
            total_user.extend(user); total_item.extend(item); total_label.extend(label)
            prediction, rating_loss = model.predict(user, item, label)
            test_prediction.extend(tensorToScalar(prediction)); test_rating_loss.extend(tensorToScalar(rating_loss))
            #import pdb; pdb.set_trace()
        t3 = time()

        val_rating_loss, test_rating_loss = np.sqrt(np.mean(val_rating_loss)), np.sqrt(np.mean(test_rating_loss))
        
        print(val_rating_loss, test_rating_loss)
        import sys; sys.exit(0)
        
        #import sys; sys.exit(0)
    print("----"*20)
    print(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    print("----"*20)