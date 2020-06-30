import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time, strftime
from copy import deepcopy
from gensim.models import Word2Vec

'''
sys.path.append('/content/drive/My Drive/task/aspect_based_rs/src/amazon_musical_test/expansion_net')
import DataModule_expansion_net as data_utils
import config_expansion_net as conf
'''
import DataModule_gru as data_utils
import config_gru as conf

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
    from gru import gru
    model = gru()
    
    #model.load_state_dict(torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_clothing/train_amazon_clothing_lm_id_X7.mod'))
    model.cuda()
    
    #model.encoder.user_embedding.weight.requires_grad = False
    #model.encoder.item_embedding.weight.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_lm_id_x.py' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_gru_lm_id_X.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_gru_lm_id_X' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data)
    train_batch_sampler = data.BatchSampler(data.RandomSampler(\
        range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    review_val_dataset = data_utils.TestData(val_data)
    review_val_sampler = data.BatchSampler(data.RandomSampler(\
        range(review_val_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    review_test_dataset = data_utils.TestData(test_data)
    review_test_sampler = data.BatchSampler(data.RandomSampler(\
        range(review_test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    max_bleu = 0.0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        #import pdb; pdb.set_trace()
        train_review_loss = []
        for batch_idx_list in train_batch_sampler:
            user_list, item_list, _, review_input_list, review_output_list = train_dataset.get_batch(batch_idx_list)
            out_loss, obj = model(user_list, item_list, review_input_list, review_output_list)
            train_review_loss.extend(tensorToScalar(out_loss))
            model.zero_grad(); obj.backward(); optimizer.step()
            #import pdb; pdb.set_trace()
        t1 = time()

        # evaluate the performance of the model with following code
        model.eval()
        
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
        
        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t1-t0)))
        log.record('Train loss:{:.4f}'.format(np.mean(train_review_loss)))
        

        #import sys; sys.exit()
    log.record("----"*20)
    log.record(f"{now()} {conf.data_name}best epoch: {review_best_epoch}")
    log.record("----"*20)