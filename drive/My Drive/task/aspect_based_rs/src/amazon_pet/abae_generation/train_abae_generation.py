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

    model_params = model.state_dict()
    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))
    for idx in range(1):
        model_params['encoder_2.word_embedding.weight'][idx] = torch.zeros(conf.abae_word_dim)
    for idx in range(1, conf.abae_vocab_sz):
        model_params['encoder_2.word_embedding.weight'][idx] = torch.FloatTensor(word_embedding.wv[word_embedding.wv.index2entity[idx-1]])

    k_means_weight = np.load('%s/%s.k_means_15.npy' % (conf.target_path, conf.data_name)) # (aspect_dimesion, word_dimension)
    model_params['encoder_2.transform_T.weight'] = torch.FloatTensor(k_means_weight.transpose()) # (word_dim,  asp_dim)
    
    model.load_state_dict(model_params)

    model.encoder_2.word_embedding.weight.requires_grad = False
    model.encoder_2.transform_T.weight.requires_grad = False

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data, test_data, user_seq_dict, item_seq_dict = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### FIRST TRAINING #####################################
    check_dir('%s/train_%s_expansion_net_id_x.log' % (conf.out_path, conf.data_name))
    log = Logging('%s/train_%s_expansion_net_id_X4.py' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_expansion_net_id_X4' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data, user_seq_dict, item_seq_dict)
    train_batch_sampler = data.BatchSampler(data.RandomSampler(\
        range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    train_dataset.construct_aspect_voab()
    aspect_count = train_dataset.count_aspect_words()
    #log.record('The number of the words which are aspect words is:%d' % aspect_count)

    review_aspect_index, review_aspect_value = train_dataset.construct_aspect_voab()
    review_aspect_mask = torch.sparse.FloatTensor(review_aspect_index.t(), \
        review_aspect_value, torch.Size([conf.gen_vocab_sz, conf.aspect_dim]))

    # prepare data for the evaluation
    review_val_dataset = data_utils.TestData(val_data, user_seq_dict, item_seq_dict)
    review_val_sampler = data.BatchSampler(data.SequentialSampler(\
        range(review_val_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    review_test_dataset = data_utils.TestData(test_data, user_seq_dict, item_seq_dict)
    review_test_sampler = data.BatchSampler(data.SequentialSampler(\
        range(review_test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    max_bleu = 0.0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()

        model.train()

        train_review_loss = []
        for batch_idx_list in train_batch_sampler:
            
            user_list, item_list, _, review_input_list, review_output_list, \
                user_pos_sent, user_neg_sent, item_pos_sent, \
                item_neg_sent =\
                train_dataset.get_batch(batch_idx_list)

            out_loss, obj = model(user_list, item_list, review_input_list, \
                review_output_list, review_aspect_mask, user_pos_sent, \
                user_neg_sent, item_pos_sent, item_neg_sent)

            train_review_loss.extend(tensorToScalar(out_loss))
            model.zero_grad(); obj.backward(); optimizer.step()
        t1 = time()

        log.record('Training Stage: Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t1-t0)))
        log.record('Train loss:{:.4f}'.format(np.mean(train_review_loss)))

        # evaluate the performance of the model with following code
        model.eval()
        
        if epoch % 5 == 0:
            val_bleu_4, rouge_L_f = evaluate(review_val_dataset, \
                review_val_sampler, model, review_aspect_mask)
        
            if (val_bleu_4+rouge_L_f) > max_bleu:
                torch.save(model.state_dict(), '%s_%d.mod' % (train_model_path, epoch))
                best_epoch = epoch
            max_bleu = max(max_bleu, (val_bleu_4+rouge_L_f))

            t2 = time()
            log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t2-t1)))
            log.record('Val: BLEU_4:%.4f, ROUGE_L_F:%.4f' % (val_bleu_4, rouge_L_f))

            t3 = time()
            test_bleu_4, test_rouge_L_f = evaluate(review_test_dataset, \
                review_test_sampler, model, review_aspect_mask)
            log.record('Epoch:{}, compute loss cost:{:.4f}s'.format(epoch, (t3-t2)))
            log.record('Test: BLEU_4:%.4f, ROUGE_L_F:%.4f' % (test_bleu_4, test_rouge_L_f))
        
    log.record("----"*20)
    log.record(f"{now()} {conf.data_name}best epoch: {best_epoch}")
    log.record("----"*20)