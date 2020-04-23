# ABAE + Rating Prediction

import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time, strftime
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
    
    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(3):
        model_params['word_embedding.weight'][idx] = torch.zeros(conf.word_dimension)
    for idx in range(3, conf.vocab_sz):
        model_params['word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-3]])
    
    k_means_weight = np.load('%s/%s.k_means.npy' % (conf.target_path, conf.data_name))
    model_params['transform_T.weight'] = torch.FloatTensor(k_means_weight.transpose()) # (aspect_dimesion, word_dimension)

    model.load_state_dict(model_params)

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data, \
        train_user_historical_review_dict, train_item_historical_review_dict = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_aspect_generation_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_aspect_generation_id_02.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_aspect_generation_id_02.mod' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data, train_user_historical_review_dict, train_item_historical_review_dict, train_data)
    val_dataset = data_utils.TrainData(val_data, train_user_historical_review_dict, train_item_historical_review_dict, train_data)
    test_dataset = data_utils.TrainData(test_data, train_user_historical_review_dict, train_item_historical_review_dict, train_data)

    train_batch_sampler = data.BatchSampler(data.RandomSampler(range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(data.RandomSampler(range(val_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    test_batch_sampler = data.BatchSampler(data.RandomSampler(range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)


    # Start Training !!!
    min_rating_loss = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()

        train_dataset.construct_aspect_voab()
        aspect_count = train_dataset.count_aspect_words()
        log.record('The number of the words which are aspect words is:%d' % aspect_count)

        review_aspect, review_aspect_bool = train_dataset.construct_aspect_voab()

        model.train()

        train_rating_loss, train_abae_loss, train_prediction, train_generation_loss = [], [], [], []
        for batch_idx_list in train_batch_sampler:
            user, item, label, pos_review, neg_review, user_idx_list, \
                item_idx_list, review_input, review_output = train_dataset.get_batch(batch_idx_list)
            
            '''
            obj, rating_loss, abae_loss, prediction = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)

            generation_loss = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)
            
            '''

            obj, rating_loss, abae_loss, prediction, generation_loss = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)

            train_rating_loss.extend(tensorToScalar(rating_loss))
            train_prediction.extend(tensorToScalar(prediction))
            train_abae_loss.extend(tensorToScalar(abae_loss))

            train_generation_loss.extend([generation_loss.item()]*len(batch_idx_list))

            model.zero_grad(); obj.backward(); optimizer.step()

        t1 = time()
        
        scheduler.step(epoch)
        
        # evaluate the performance of the model with following code
        model.eval()

        val_rating_loss, val_prediction, val_generation_loss = [], [], []
        for batch_idx_list in val_batch_sampler:
            user, item, label, pos_review, neg_review, user_idx_list, \
                item_idx_list, review_input, review_output = val_dataset.get_batch(batch_idx_list)
            
            '''
            obj, rating_loss, abae_loss, prediction = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)
            
            generation_loss = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)
            
            '''

            obj, rating_loss, abae_loss, prediction, generation_loss = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)

            val_prediction.extend(tensorToScalar(prediction))
            val_rating_loss.extend(tensorToScalar(rating_loss))

            val_generation_loss.extend([generation_loss.item()]*len(batch_idx_list))
            
        t2 = time()

        test_rating_loss, test_prediction, test_generation_loss = [], [], []
        for batch_idx_list in test_batch_sampler:
            user, item, label, pos_review, neg_review, user_idx_list, \
                item_idx_list, review_input, review_output = test_dataset.get_batch(batch_idx_list)

            '''
            obj, rating_loss, abae_loss, prediction = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)

            generation_loss = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)
            
            '''

            obj, rating_loss, abae_loss, prediction, generation_loss = model(pos_review, neg_review, user, item, label, \
                user_idx_list, item_idx_list, review_input, review_output, review_aspect, review_aspect_bool)

            test_prediction.extend(tensorToScalar(prediction))
            test_rating_loss.extend(tensorToScalar(rating_loss))

            test_generation_loss.extend([generation_loss.item()]*len(batch_idx_list))

        t3 = time()

        train_rating_rmse, val_rating_rmse, test_rating_rmse = np.sqrt(np.mean(train_rating_loss)), \
            np.sqrt(np.mean(val_rating_loss)), np.sqrt(np.mean(test_rating_loss))
        train_generation_loss, val_generation_loss, test_generation_loss = np.mean(train_generation_loss),\
            np.mean(val_generation_loss), np.mean(test_generation_loss)


        if epoch == 1:
            min_rating_loss = val_rating_rmse
            min_review_loss = val_generation_loss
        if val_rating_rmse <= min_rating_loss:
            torch.save(model.state_dict(), '%s_rating' % train_model_path)
            log.record('-----------save rating model------------')
            rating_best_epoch = epoch
        if val_generation_loss <= min_review_loss:
            torch.save(model.state_dict(), '%s_review' % train_model_path)
            log.record('-----------save review model------------')
            review_best_epoch = epoch
        min_rating_loss = min(val_rating_rmse, min_rating_loss)
        min_review_loss = min(val_generation_loss, min_review_loss)

        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t1-t0)))
        log.record('ABAE loss:{:.4f}'.format(np.mean(train_abae_loss)))
        log.record('Rating RMSE: Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_rating_rmse, val_rating_rmse, test_rating_rmse))

        log.record('Review NLL: Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_generation_loss, val_generation_loss, test_generation_loss))

        log.record('Train prediction mean:%.4f, var:%.4f' % (np.mean(train_prediction), np.var(train_prediction)))
        log.record('Val prediction mean:%.4f, var:%.4f' % (np.mean(val_prediction), np.var(val_prediction)))
        log.record('Test prediction mean:%.4f, var:%.4f' % (np.mean(test_prediction), np.var(test_prediction)))

    log.record("----"*20)
    log.record(f"{now()} {conf.data_name}  best rating epoch: {rating_best_epoch}")
    log.record(f"{now()} {conf.data_name}  best review epoch: {review_best_epoch}")
    log.record("----"*20)