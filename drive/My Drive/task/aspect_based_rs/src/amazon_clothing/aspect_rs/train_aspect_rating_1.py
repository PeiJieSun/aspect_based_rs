import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time, strftime
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_aspect as data_utils
import config_aspect as conf

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
    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data, \
        train_user_historical_review_dict, train_item_historical_review_dict = data_utils.load_all()
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
    
    #model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_aspect_rating_1_id_adabound_19.mod'))

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    #import adabound
    #optimizer = adabound.AdaBound(model.parameters(), lr=conf.learning_rate, final_lr=0.1)

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_aspect_rating_1_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_aspect_rating_1_id_adabound_20.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_aspect_rating_1_id_adabound_20.mod' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data, train_user_historical_review_dict, train_item_historical_review_dict)
    val_dataset = data_utils.ValData(val_data)
    test_dataset = data_utils.ValData(test_data)

    train_batch_sampler = data.BatchSampler(data.RandomSampler(range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(data.SequentialSampler(range(val_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    test_batch_sampler = data.BatchSampler(data.SequentialSampler(range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    user_aspect_embedding = np.random.rand(conf.num_users, conf.common_dimension)
    item_aspect_embedding = np.random.rand(conf.num_items, conf.common_dimension)

    # Start Training !!!
    min_rating_loss = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        total_user, total_label = [], []
        train_rating_loss, train_abae_loss, train_prediction = [], [], []
        for batch_idx_list in train_batch_sampler:
            user_list, item_list, rating_list, review_input_list, \
                neg_review, user_histor_index, user_histor_value, \
                item_histor_index, item_histor_value = train_dataset.get_batch(batch_idx_list)

            total_user.extend(user_list); total_label.extend(rating_list)

            obj, rating_loss, abae_loss, prediction, user_aspect_embed, item_aspect_embed = \
                model(review_input_list, neg_review, \
                user_list, item_list, rating_list, user_histor_index, user_histor_value, item_histor_index, item_histor_value)
            #import pdb; pdb.set_trace()
            train_rating_loss.extend(tensorToScalar(rating_loss)); train_abae_loss.extend(tensorToScalar(abae_loss))
            train_prediction.extend(tensorToScalar(prediction))
            model.zero_grad(); obj.backward(); optimizer.step()

            for idx, user in enumerate(user_list):
                user_aspect_embedding[user] = tensorToScalar(user_aspect_embed[idx])
            for idx, item in enumerate(item_list):
                item_aspect_embedding[item] = tensorToScalar(item_aspect_embed[idx])

        t1 = time()

        
        scheduler.step(epoch)

        # Update user_embedding & item_embedding with generated aspect-based user&item embedding
        model.user_embedding.weight = nn.Parameter(torch.FloatTensor(user_aspect_embedding).cuda())
        model.item_embedding.weight = nn.Parameter(torch.FloatTensor(item_aspect_embedding).cuda())

        #import pdb; pdb.set_trace()

        # evaluate the performance of the model with following code
        model.eval()
        
        total_user, total_label = [], []
        val_rating_loss, val_prediction = [], []
        for batch_idx_list in val_batch_sampler:
            user_list, item_list, rating_list = val_dataset.get_batch(batch_idx_list)
            total_user.extend(user_list); total_label.extend(rating_list)
            prediction, rating_loss = model.predict(user_list, item_list, rating_list)
            val_prediction.extend(tensorToScalar(prediction)); val_rating_loss.extend(tensorToScalar(rating_loss))
        t2 = time()


        total_user, total_item, total_label = [], [], []
        test_rating_loss, test_prediction = [], []
        for batch_idx_list in test_batch_sampler:
            user_list, item_list, rating_list = test_dataset.get_batch(batch_idx_list)
            total_user.extend(user_list); total_item.extend(item_list); total_label.extend(rating_list)
            prediction, rating_loss = model.predict(user_list, item_list, rating_list)
            test_prediction.extend(tensorToScalar(prediction)); test_rating_loss.extend(tensorToScalar(rating_loss))
            #import pdb; pdb.set_trace()
        t3 = time()
        
        if epoch == 1:
            min_rating_loss = np.sqrt(np.mean(val_rating_loss))
        if np.sqrt(np.mean(val_rating_loss)) < min_rating_loss:
            torch.save(model.state_dict(), train_model_path)
            print('save model')
            best_epoch = epoch
        min_rating_loss = min(np.sqrt(np.mean(val_rating_loss)), min_rating_loss)

        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t1-t0)))
        log.record('ABAE: Train loss:{:.4f}'.format(np.mean(train_abae_loss)))
        log.record('Rating RMSE: Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(
            np.sqrt(np.mean(train_rating_loss)), np.sqrt(np.mean(val_rating_loss)), np.sqrt(np.mean(test_rating_loss))))

        log.record('Train prediction mean:%.4f, var:%.4f' % (np.mean(train_prediction), np.var(train_prediction)))
        log.record('Val prediction mean:%.4f, var:%.4f' % (np.mean(val_prediction), np.var(val_prediction)))
        log.record('Test prediction mean:%.4f, var:%.4f' % (np.mean(test_prediction), np.var(test_prediction)))

        log.record('user embedding mean:%.4f, var:%.4f' % \
            (torch.mean(model.user_embedding.weight).item(), torch.var(model.user_embedding.weight).item()))
        log.record('item embedding mean:%.4f, var:%.4f' % \
            (torch.mean(model.item_embedding.weight).item(), torch.var(model.item_embedding.weight).item()))

        #import sys; sys.exit(0)

    print("----"*20)
    print(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    print("----"*20)