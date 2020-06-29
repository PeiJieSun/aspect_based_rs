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
    from mrg import mrg
    model = mrg()

    #model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_clothing/train_amazon_clothing_lm_id_X7.mod'))
    model.cuda()
    
    #model.encoder.user_embedding.weight.requires_grad = False
    #model.encoder.item_embedding.weight.requires_grad = False

    review_optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    rating_optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_lm_id_x.py' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_mrg_id_X.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_mrg_id_X' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data)
    train_batch_sampler = data.BatchSampler(data.RandomSampler(\
        range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    val_dataset = data_utils.TrainData(val_data)
    val_batch_sampler = data.BatchSampler(data.RandomSampler(\
        range(val_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    test_dataset = data_utils.TrainData(test_data)
    test_batch_sampler = data.BatchSampler(data.RandomSampler(\
        range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # construct review validate dataset
    review_val_dataset = data_utils.TestData(val_data)
    review_val_sampler = data.BatchSampler(data.RandomSampler(\
        range(review_val_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    review_test_dataset =  data_utils.TestData(test_data)
    review_test_sampler = data.BatchSampler(data.RandomSampler(\
        range(review_test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    max_bleu = 0.0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        #import pdb; pdb.set_trace()
        train_rating_loss, train_review_loss = [], []
        for batch_idx_list in train_batch_sampler:
            user_list, item_list, label_list, review_input_list, \
                review_output_list = train_dataset.get_batch(batch_idx_list)
            rating_out_loss, review_out_loss, obj = model(user_list, item_list, \
                label_list, review_input_list, review_output_list)

            model.zero_grad(); obj.backward(); review_optimizer.step()

            train_rating_loss.extend(tensorToScalar(rating_out_loss))
            train_review_loss.extend(tensorToScalar(review_out_loss))
        t1 = time()

        # evaluate the performance of the model with following code
        model.eval()

        val_rating_loss = []
        for batch_idx_list in val_batch_sampler:
            user_list, item_list, label_list, _, _ =\
                val_dataset.get_batch(batch_idx_list)

            rating_out_loss = model.predict_rating(user_list, item_list, label_list)
            val_rating_loss.extend(tensorToScalar(rating_out_loss))

        test_rating_loss = []
        for batch_idx_list in test_batch_sampler:
            user_list, item_list, label_list, _, _ =\
                test_dataset.get_batch(batch_idx_list)

            rating_out_loss = model.predict_rating(user_list, item_list, label_list)
            test_rating_loss.extend(tensorToScalar(rating_out_loss))

        train_rating_loss, val_rating_loss, test_rating_loss = \
            np.sqrt(np.mean(train_rating_loss)), np.sqrt(np.mean(val_rating_loss)), np.sqrt(np.mean(test_rating_loss))

        if epoch == 1:
            min_rating_loss = val_rating_loss
        if val_rating_loss < min_rating_loss:
            #torch.save(model.state_dict(), train_model_path)
            rating_best_epoch = epoch
        min_rating_loss = min(val_rating_loss, min_rating_loss)

        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t1-t0)))
        log.record('NLL Train loss:{:.4f}'.format(np.mean(train_review_loss)))
        log.record('RMSE Train loss:{:.4f}, Val loss:{:.4f}, Test loss:{:.4f}'.format(\
            train_rating_loss, val_rating_loss, test_rating_loss))

        if epoch % 5 == 0:
            val_bleu_4, val_rouge_L_f = evaluate(review_val_dataset, review_val_sampler, model)
            if (val_bleu_4+val_rouge_L_f) > max_bleu:
                torch.save(model.state_dict(), '%s_%d.mod' % (train_model_path, epoch))
                review_best_epoch = epoch
            max_bleu = max(max_bleu, (val_bleu_4+val_rouge_L_f))

            t2 = time()
            log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t2-t1)))
            log.record('Val: BLEU_4:%.4f, ROUGE_L_F:%.4f' % (val_bleu_4, val_rouge_L_f))

            t3 = time()
            test_bleu_4, test_rouge_L_f = evaluate(review_test_dataset, review_test_sampler, model)
            log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t2)))
            log.record('Test: BLEU_4:%.4f, ROUGE_L_F:%.4f' % (test_bleu_4, test_rouge_L_f))


    log.record("----"*20)
    log.record(f"{now()} {conf.data_name} rating best epoch: {rating_best_epoch}")
    log.record(f"{now()} {conf.data_name} review best epoch: {review_best_epoch}")
    log.record("----"*20)