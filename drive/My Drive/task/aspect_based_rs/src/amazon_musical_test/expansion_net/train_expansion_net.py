import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time, strftime
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_expansion_net as data_utils
import config_expansion_net as conf

from evaluate import evaluate

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
    from expansion_net import expansion_net
    model = expansion_net()
    
    #model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/model/train_amazon_clothing_expansion_net_id_33.mod'))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_expansion_net_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_expansion_net_id_X2.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_expansion_net_id_X2' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data)
    train_batch_sampler = data.BatchSampler(\
        data.RandomSampler(range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    val_dataset = data_utils.TestData(val_data)
    val_batch_sampler = data.BatchSampler(\
        data.SequentialSampler(range(val_dataset.length)), batch_size=1, drop_last=False)

    # Start Training !!!
    max_bleu = 0.0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()

        train_dataset.construct_aspect_voab()
        aspect_count = train_dataset.count_aspect_words()
        #log.record('The number of the words which are aspect words is:%d' % aspect_count)

        review_aspect, review_aspect_mask = train_dataset.construct_aspect_voab()

        model.train()

        train_loss = []
        train_ref, train_hyp = [], []
        for batch_idx_list in train_batch_sampler:
            user, item, label, review_input, review_output, summary = \
                train_dataset.get_batch(batch_idx_list)
            generation_loss = model(user, item, summary, review_input, \
                review_output, review_aspect, review_aspect_mask)
            train_loss.append(generation_loss.item())
            model.zero_grad(); generation_loss.backward(); optimizer.step()
        t1 = time()
        train_loss = np.mean(train_loss)
        
        log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t1-t0)))
        log.record('Train:{:.4f}'.format(train_loss))

        # evaluate the performance of the model with following xxx 
        model.eval()
        
        if epoch % 5 == 0:
            val_bleu_4, rouge_L_f = evaluate(val_dataset, val_batch_sampler, \
                model, review_aspect, review_aspect_mask)
            torch.save(model.state_dict(), '%s_%d.mod' % (train_model_path, epoch))

            if val_bleu_4 > max_bleu:
                best_epoch = epoch
            max_bleu = max(max_bleu, val_bleu_4)

            t2 = time()
            log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t2-t1)))
            log.record('Val: BLEU_4:%.4f, ROUGE_L_F:%.4f' % (val_bleu_4, rouge_L_f))

    log.record("----"*20)
    log.record(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    log.record("----"*20)