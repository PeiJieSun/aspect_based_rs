import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time, strftime
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_mrg as data_utils
import config_mrg as conf

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
    from mrg import mrg
    model = mrg()

    # load word embedding from pretrained word2vec model
    model_params = model.state_dict()

    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(3):
        model_params['word_embedding.weight'][idx] = torch.zeros(conf.word_dimension)
    for idx in range(3, conf.vocab_sz):
        model_params['word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-3]])

    model.load_state_dict(model_params)

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
    check_dir('%s/train_%s_mrg_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_mrg_id_06.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_mrg_id_06.mod' % (conf.out_path, conf.data_name)

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
        
        train_rating_loss, train_generation_loss, train_prediction = [], [], []
        for batch_idx_list in val_batch_sampler:
            user, item, label, review_input, review_output = train_dataset.get_batch(batch_idx_list)
            prediction, rating_loss, generation_loss, obj_loss = model(user, item, label, review_input, review_output)
            
            train_rating_loss.extend([rating_loss.item()]*len(batch_idx_list))
            train_generation_loss.extend([generation_loss.item()]*len(batch_idx_list))
            train_prediction.extend(tensorToScalar(prediction))

            model.zero_grad(); obj_loss.backward(); optimizer.step()
        t2 = time()

        # evaluate the performance of the model with following xxx 
        model.eval()
        
        val_rating_loss, val_generation_loss, val_prediction = [], [], []
        for batch_idx_list in val_batch_sampler:
            user, item, label, review_input, review_output = val_dataset.get_batch(batch_idx_list)
            prediction, rating_loss, generation_loss, obj_loss = model(user, item, label, review_input, review_output)
            
            val_rating_loss.extend([rating_loss.item()]*len(batch_idx_list))
            val_generation_loss.extend([generation_loss.item()]*len(batch_idx_list))
            val_prediction.extend(tensorToScalar(prediction))

        t2 = time()

        test_rating_loss, test_generation_loss, test_prediction = [], [], []
        for batch_idx_list in test_batch_sampler:
            user, item, label, review_input, review_output = test_dataset.get_batch(batch_idx_list)
            prediction, rating_loss, generation_loss, obj_loss = model(user, item, label, review_input, review_output)
            
            test_rating_loss.extend([rating_loss.item()]*len(batch_idx_list))
            test_generation_loss.extend([generation_loss.item()]*len(batch_idx_list))
            test_prediction.extend(tensorToScalar(prediction))

        t3 = time()
        
        train_rating_loss, val_rating_loss, test_rating_loss =\
            np.mean(train_rating_loss), np.mean(val_rating_loss), np.mean(test_rating_loss)
        train_generation_loss, val_generation_loss, test_generation_loss =\
            np.mean(train_generation_loss), np.mean(val_generation_loss), np.mean(test_generation_loss)

        if epoch == 1:
            min_loss = val_rating_loss
        if val_rating_loss <= min_loss:
            torch.save(model.state_dict(), train_model_path)
            best_epoch = epoch
        min_loss = min(min_loss, val_rating_loss)
        
        log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('Review generation: Train:{:.4f}, Val:{:.4f}, Test:{:.4f}'.format(train_rating_loss, val_rating_loss, test_rating_loss))
        log.record('Rating prediction: Train:{:.4f}, Val:{:.4f}, Test:{:.4f}'.format(train_generation_loss, val_generation_loss, test_generation_loss))
        log.record('train prediction mean:%.4f, var:%.4f' % (np.mean(train_prediction), np.var(train_prediction)))
        log.record('val prediction mean:%.4f, var:%.4f' % (np.mean(val_prediction), np.var(val_prediction)))
        log.record('test prediction mean:%.4f, var:%.4f' % (np.mean(test_prediction), np.var(test_prediction)))

        #import sys; sys.exit(0)
    print("----"*20)
    print(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    print("----"*20)