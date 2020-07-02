import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time, strftime
from copy import deepcopy
from gensim.models import Word2Vec

model_name = 'mrg'
sys.path.append('/content/drive/My Drive/task/aspect_based_rs/src/amazon_musical_test/%s' % model_name)

exec('import DataModule_%s as data_utils' % model_name)
exec('import config_%s as conf' % model_name)

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
    data_set = data_utils.load_all()
    train_data, val_data, test_data = data_set[0], data_set[1], data_set[2]
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    ############################## CREATE MODEL ##############################

    model_name = 'probe'
    exec('from %s import %s' % (model_name, model_name))
    exec('model = %s()' % model_name)

    #import pdb; pdb.set_trace()
    
    #model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_clothing/train_amazon_clothing_pmf_id_X1.mod'))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)


    ########################### FIRST TRAINING #####################################
    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data)
    train_batch_sampler = data.BatchSampler(data.RandomSampler(\
        range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    val_dataset = data_utils.TrainData(val_data)
    val_batch_sampler = data.BatchSampler(data.SequentialSampler(\
        range(val_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    test_dataset = data_utils.TrainData(test_data)
    test_batch_sampler = data.BatchSampler(data.SequentialSampler(\
        range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    min_loss = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        train_loss, train_prediction = [], []
        for batch_idx_list in train_batch_sampler:
            values = train_dataset.get_batch(batch_idx_list)

            prediction, mse_loss, obj_loss = model(values)
            #import pdb; pdb.set_trace()
            train_loss.extend(tensorToScalar(mse_loss))
            train_prediction.extend(tensorToScalar(prediction))

            model.zero_grad(); obj_loss.backward(); optimizer.step()
        t1 = time()

        #import pdb; pdb.set_trace()
        # evaluate the performance of the model with following xxx 
        model.eval()
        
        val_loss, val_prediction = [], []
        for batch_idx_list in val_batch_sampler:
            values = val_dataset.get_batch(batch_idx_list)

            prediction, mse_loss, _ = model(values)    
            val_loss.extend(tensorToScalar(mse_loss))
            val_prediction.extend(tensorToScalar(prediction))
        t2 = time()

        test_loss, test_prediction = [], []
        for batch_idx_list in test_batch_sampler:
            values = test_dataset.get_batch(batch_idx_list)

            prediction, mse_loss, _ = model(values)    
            test_loss.extend(tensorToScalar(mse_loss))
            test_prediction.extend(tensorToScalar(prediction))
        t3 = time()

        train_loss, val_loss, test_loss = np.sqrt(np.mean(train_loss)), np.sqrt(np.mean(val_loss)), np.sqrt(np.mean(test_loss))

        if epoch == 1:
            min_rating_loss = val_loss
        if val_loss < min_rating_loss:
            best_epoch = epoch
        min_rating_loss = min(val_loss, min_rating_loss)
        
        print('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t0)))
        print('Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(train_loss, val_loss, test_loss))

        print('Train prediction mean:%.4f, var:%.4f' % (np.mean(train_prediction), np.var(train_prediction)))
        print('Val prediction mean:%.4f, var:%.4f' % (np.mean(val_prediction), np.var(val_prediction)))
        print('Test prediction mean:%.4f, var:%.4f' % (np.mean(test_prediction), np.var(test_prediction)))

        '''
        print('user embedding mean:%.4f, var:%.4f' % \
            (torch.mean(model.embedding_user.weight).item(), torch.var(model.embedding_user.weight).item()))
        print('item embedding mean:%.4f, var:%.4f' % \
            (torch.mean(model.embedding_item.weight).item(), torch.var(model.embedding_item.weight).item()))
        '''

    print("----"*20)
    print(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    print("----"*20)