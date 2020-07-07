import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time, strftime
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_abae_rs as data_utils
import config_abae_rs as conf

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
    from abae_rs import abae_rs
    model = abae_rs()

    model_params = model.state_dict()
    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(1):
        model_params['encoder.word_embedding.weight'][idx] = torch.zeros(conf.word_dim)
    for idx in range(1, conf.vocab_sz):
        model_params['encoder.word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-1]])

    k_means_weight = np.load('%s/%s.k_means_15.npy' % (conf.target_path, conf.data_name)) # (aspect_dimesion, word_dimension)
    model_params['encoder.transform_T.weight'] = torch.FloatTensor(k_means_weight.transpose()) # (word_dim,  asp_dim)
    
    model.load_state_dict(model_params)

    model.encoder.word_embedding.weight.requires_grad = False
    #model.encoder.transform_T.weight.requires_grad = False

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data, user_seq_dict, item_seq_dict = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_abae_rs_id_x.py' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_abae_rs_id_X.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_abae_rs_id_X.mod' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data, user_seq_dict, item_seq_dict)
    val_dataset = data_utils.TrainData(val_data, user_seq_dict, item_seq_dict)
    test_dataset = data_utils.TrainData(test_data, user_seq_dict, item_seq_dict)

    train_batch_sampler = data.BatchSampler(\
        data.RandomSampler(range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(\
        data.RandomSampler(range(val_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    test_batch_sampler = data.BatchSampler(\
        data.RandomSampler(range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    min_loss = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        train_pred, val_pred, test_pred = [], [], []

        train_loss = []
        for batch_idx_list in train_batch_sampler:
            user, item, label, user_pos_sent, user_neg_sent, item_pos_sent, \
                item_neg_sent = train_dataset.get_batch(batch_idx_list)
            rating_pred, rating_obj_loss, rating_out_loss = \
                model(user, item, label, user_pos_sent, user_neg_sent, item_pos_sent, item_neg_sent)

            train_pred.extend(tensorToScalar(rating_pred))

            train_loss.extend(tensorToScalar(rating_out_loss))
            model.zero_grad(); rating_obj_loss.backward(); optimizer.step()
        t1 = time()

        # evaluate the performance of the model with following code
        model.eval()
        
        val_loss = []
        for batch_idx_list in val_batch_sampler:
            user, item, label, user_pos_sent, user_neg_sent, item_pos_sent, \
                item_neg_sent = val_dataset.get_batch(batch_idx_list)
            rating_pred, rating_obj_loss, rating_out_loss = \
                model(user, item, label, user_pos_sent, user_neg_sent, item_pos_sent, item_neg_sent)
            
            val_pred.extend(tensorToScalar(rating_pred))

            val_loss.extend(tensorToScalar(rating_out_loss))
        t2 = time()

        test_loss = []
        for batch_idx_list in test_batch_sampler:
            user, item, label, user_pos_sent, user_neg_sent, item_pos_sent, \
                item_neg_sent = test_dataset.get_batch(batch_idx_list)
            rating_pred, rating_obj_loss, rating_out_loss = \
                model(user, item, label, user_pos_sent, user_neg_sent, item_pos_sent, item_neg_sent)
            
            test_pred.extend(tensorToScalar(rating_pred))

            test_loss.extend(tensorToScalar(rating_out_loss))
        t3 = time()

        train_loss, val_loss, test_loss = np.sqrt(np.mean(train_loss)), \
            np.sqrt(np.mean(val_loss)), np.sqrt(np.mean(test_loss))

        if epoch == 1:
            min_loss = val_loss
        if val_loss < min_loss:
            #torch.save(model.state_dict(), train_model_path)
            best_epoch = epoch
        min_loss = min(val_loss, min_loss)

        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_loss, val_loss, test_loss))

        log.record('Train prediction mean:%.4f, var:%.4f' % (np.mean(train_pred), np.var(train_pred)))
        log.record('Val prediction mean:%.4f, var:%.4f' % (np.mean(val_pred), np.var(val_pred)))
        log.record('Test prediction mean:%.4f, var:%.4f' % (np.mean(test_pred), np.var(test_pred)))

    log.record("----"*20)
    log.record(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    log.record("----"*20)