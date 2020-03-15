import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_aspect as data_utils
import config_aspect as conf

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
    train_data, val_data, test_data, train_review_embedding, val_review_embedding, test_review_embedding,\
    train_user_historical_review_dict, train_item_historical_review_dict,\
    val_user_historical_review_dict, val_item_historical_review_dict,\
    test_user_historical_review_dict, test_item_historical_review_dict = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    ############################## CREATE MODEL ##############################
    from aspect_rating_1 import aspect_rating_1
    model = aspect_rating_1()
    
    model_params = model.state_dict()
    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(3):
        model_params['word_embedding.weight'][idx] = torch.zeros(conf.word_dimension)
    for idx in range(3, conf.vocab_sz):
        model_params['word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-3]])
    
    k_means_weight = np.load('%s/%s.k_means.npy' % (conf.target_path, conf.data_name))
    model_params['transform_T.weight'] = torch.FloatTensor(k_means_weight.transpose()) # (aspect_dimesion, word_dimension)
    
    model.load_state_dict(model_params)
    
    #model.load_state_dict(torch.load('%s/train_%s_aspect_rating_1_id_adabound.mod' % (conf.model_path, conf.data_name)))
    #model.load_state_dict(torch.load('%s/train_%s_abae_id_adabound.mod' % (conf.model_path, conf.data_name)))
    
    model.cuda()
    #optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    import adabound
    optimizer = adabound.AdaBound(model.parameters(), lr=conf.learning_rate, final_lr=0.1)

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_aspect_rating_1_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_aspect_rating_1_id_adabound_x3.log' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_aspect_rating_1_id_adabound_x3.mod' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data, train_review_embedding, train_user_historical_review_dict, train_item_historical_review_dict)
    val_dataset = data_utils.TrainData(val_data, val_review_embedding, val_user_historical_review_dict, val_item_historical_review_dict)
    test_dataset = data_utils.TrainData(test_data, test_review_embedding, test_user_historical_review_dict, test_item_historical_review_dict)

    train_batch_sampler = data.BatchSampler(data.RandomSampler(range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(data.RandomSampler(range(val_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    test_batch_sampler = data.BatchSampler(data.RandomSampler(range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    min_rating_loss, min_abae_loss = 0, 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        train_rating_loss, train_abae_loss, train_prediction = [], [], []
        for batch_idx_list in train_batch_sampler:
            user_list, item_list, rating_list, review_input_list, review_pos_embedding, \
                review_neg_embedding, user_histor_index, user_histor_value, \
                item_histor_index, item_histor_value = train_dataset.get_batch(batch_idx_list)
            obj, rating_loss, abae_loss, prediction = model(review_input_list, review_pos_embedding, review_neg_embedding, \
                user_list, item_list, rating_list, user_histor_index, user_histor_value, item_histor_index, item_histor_value)
            train_rating_loss.extend(tensorToScalar(rating_loss)); train_abae_loss.extend(tensorToScalar(abae_loss))
            train_prediction.extend(tensorToScalar(prediction))
            model.zero_grad(); obj.backward(); optimizer.step()
        t1 = time()
        
        # evaluate the performance of the model with following code
        model.eval()
        
        val_rating_loss, val_abae_loss, val_prediction = [], [], []
        for batch_idx_list in val_batch_sampler:
            user_list, item_list, rating_list, review_input_list, review_pos_embedding, \
                review_neg_embedding, user_histor_index, user_histor_value, \
                item_histor_index, item_histor_value = val_dataset.get_batch(batch_idx_list)
            _, rating_loss, abae_loss, prediction = model(review_input_list, review_pos_embedding, review_neg_embedding, \
                user_list, item_list, rating_list, user_histor_index, user_histor_value, item_histor_index, item_histor_value)
            val_rating_loss.extend(tensorToScalar(rating_loss)); val_abae_loss.extend(tensorToScalar(abae_loss))
            val_prediction.extend(tensorToScalar(prediction))
        t2 = time()

        test_rating_loss, test_abae_loss, test_prediction = [], [], []
        for batch_idx_list in test_batch_sampler:
            user_list, item_list, rating_list, review_input_list, review_pos_embedding, \
                review_neg_embedding, user_histor_index, user_histor_value, \
                item_histor_index, item_histor_value = test_dataset.get_batch(batch_idx_list)
            _, rating_loss, abae_loss, prediction = model(review_input_list, review_pos_embedding, review_neg_embedding, \
                user_list, item_list, rating_list, user_histor_index, user_histor_value, item_histor_index, item_histor_value)
            test_rating_loss.extend(tensorToScalar(rating_loss)); test_abae_loss.extend(tensorToScalar(abae_loss))
            test_prediction.extend(tensorToScalar(prediction))
        t3 = time()
        
        if epoch == 1:
            min_rating_loss = val_rating_loss
        if val_rating_loss < min_rating_loss:
            torch.save(model.state_dict(), train_model_path)
        min_rating_loss = min(val_rating_loss, min_rating_loss)
        
        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t1-t0)))
        log.record('Rating RMSE: Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(np.mean(train_rating_loss), np.mean(val_rating_loss), np.mean(test_rating_loss)))
        log.record('ABAE: Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(np.mean(train_abae_loss), np.mean(val_abae_loss), np.mean(test_abae_loss)))

        #import pdb; pdb.set_trace()