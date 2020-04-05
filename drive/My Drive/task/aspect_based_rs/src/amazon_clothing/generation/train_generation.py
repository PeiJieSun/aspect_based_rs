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

    # load word embedding from pretrained word2vec model
    model_params = model.state_dict()

    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(3):
        model_params['word_embedding.weight'][idx] = torch.zeros(conf.word_dimension)
    for idx in range(3, conf.vocab_sz):
        model_params['word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-3]])
    
    k_means_weight = np.load('%s/%s.k_means.npy' % (conf.target_path, conf.data_name))
    model_params['transform_T.weight'] = torch.FloatTensor(k_means_weight.transpose()) # (aspect_dimesion, word_dimension)

    '''
    aspect_rating_params = torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_aspect_rating_1_id_adabound_19.mod')
    for param in aspect_rating_params:
        if param in model_params:
            model_params[param] = aspect_rating_params[param]
            print(param)
    model.load_state_dict(model_params)
    '''

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
    
    aspect_user_embedding = np.random.rand(conf.num_users, conf.common_dimension)
    aspect_item_embedding = np.random.rand(conf.num_items, conf.common_dimension)

    # Start Training !!!
    min_rating_loss = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        total_user, total_label = [], []
        train_rating_loss, train_abae_loss, train_generation_loss, train_prediction = [], [], [], []
        for batch_idx_list in train_batch_sampler:
            user, item, label, historical_review, neg_review, \
                user_histor_index, user_histor_value, item_histor_index, \
                item_histor_value, review_input, review_output, \
                review_aspect_bool, review_aspect = train_dataset.get_batch(batch_idx_list)
            
            total_user.extend(user); total_label.extend(label)

            aspect_user_embed, aspect_item_embed, prediction, mse_loss, J_loss, \
                generation_loss, obj_loss = model(review_input, review_output, review_aspect, review_aspect_bool, \
                historical_review, neg_review, user, item, label, user_histor_index, user_histor_value, \
                item_histor_index, item_histor_value)

            train_rating_loss.extend([mse_loss.item()]*len(batch_idx_list))
            train_abae_loss.extend([J_loss.item()]*len(batch_idx_list))
            train_generation_loss.extend([generation_loss.item()]*len(batch_idx_list))
            train_prediction.extend(tensorToScalar(prediction))
            
            model.zero_grad(); obj_loss.backward(); optimizer.step()

            for idx, u in enumerate(user):
                aspect_user_embedding[u] = tensorToScalar(aspect_user_embed[idx])
            for idx, it in enumerate(item):
                aspect_item_embedding[it] = tensorToScalar(aspect_item_embed[idx])

        t1 = time()

        scheduler.step(epoch)

        train_dataset.construct_aspect_voab()

        model.aspect_user_embedding.weight = nn.Parameter(torch.FloatTensor(aspect_user_embedding).cuda())
        model.aspect_item_embedding.weight = nn.Parameter(torch.FloatTensor(aspect_item_embedding).cuda())

        # evaluate the performance of the model with following xxx 
        model.eval()
        
        #import pdb; pdb.set_trace()

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

        train_rating_loss, val_rating_loss, test_rating_loss = \
            np.sqrt(np.mean(train_rating_loss)), np.sqrt(np.mean(val_rating_loss)), np.sqrt(np.mean(test_rating_loss))
        
        
        if epoch == 1:
            min_rating_loss = val_rating_loss
        if val_rating_loss < min_rating_loss:
            torch.save(model.state_dict(), train_model_path)
            best_epoch = epoch
        min_rating_loss = min(val_rating_loss, val_rating_loss)
        

        log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        log.record('ABAE loss:%.4f, GENERATION loss:%.4f' % (np.mean(train_abae_loss), np.mean(train_generation_loss)))
        log.record('Train RMSE:{:.4f}, Val RMSE:{:.4f}, Test RMSE:{:.4f}'.format(train_rating_loss, val_rating_loss, test_rating_loss))

        log.record('Train prediction mean:%.4f, var:%.4f' % (np.mean(train_prediction), np.var(train_prediction)))
        log.record('Val prediction mean:%.4f, var:%.4f' % (np.mean(val_prediction), np.var(val_prediction)))
        log.record('Test prediction mean:%.4f, var:%.4f' % (np.mean(test_prediction), np.var(test_prediction)))

        log.record('user embedding mean:%.4f, var:%.4f' % \
            (torch.mean(model.aspect_user_embedding.weight).item(), torch.var(model.aspect_user_embedding.weight).item()))
        log.record('item embedding mean:%.4f, var:%.4f' % \
            (torch.mean(model.aspect_item_embedding.weight).item(), torch.var(model.aspect_item_embedding.weight).item()))

        #import sys; sys.exit(0)
    print("----"*20)
    print(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    print("----"*20)