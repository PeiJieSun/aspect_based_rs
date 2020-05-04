import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np
from collections import defaultdict

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_language_model as data_utils
import config_language_model as conf

from Logging import Logging

PAD = 0; SOS = 1; EOS = 2

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

class geneated_review():
    def __init__(self):
        self.previous_data = defaultdict(list)
        self.current_data = defaultdict(list)

if __name__ == '__main__':
    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    test_data = data_utils.load_test_data()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    
    ############################## LOAD MODEL ##############################
    from language_model import language_model
    model = language_model()

    model.load_state_dict(
        torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_electronics/train_amazon_electronics_language_model_id_x.mod'))

    model.cuda()

    ########################### TEST STAGE #####################################
    test_dataset = data_utils.TrainData(test_data)

    test_batch_sampler = data.BatchSampler(data.SequentialSampler(range(test_dataset.length)), batch_size=conf.batch_size, drop_last=False)

    # Start Training !!!
    t0 = time()
    model.eval()

    beam_size = 4
    previous_sequence = defaultdict(dict)
    current_sequence = defaultdict(dict)

    batch_idx_list = [0, 1, 2]
    # initialize the dictionary which stores the generated words token idx
    for idx in range(len(batch_idx_list)):
        for sub_idx in range(beam_size):
            previous_sequence[idx][sub_idx] = [SOS]

    # first iteration: prepare the input text data
    initial_word_input = torch.LongTensor([SOS, SOS, PAD]).view(1, -1).cuda() # (1, batch_size)
    batch_size = initial_word_input.shape[1]

    # construct the hidden_state for current iteration
    hidden_state = (torch.zeros(2, batch_size, conf.hidden_size).cuda(),
        torch.zeros(2, batch_size, conf.hidden_size).cuda()) # (2, batch_size, hidden_size)

    # hidden_state: ((2, batch_size, hidden_size), (2, batch_size, hidden_size))
    output_word_embedding, hidden_state = model(initial_word_input, hidden_state) # output_word_embedding: (1*batch_size, word_dimension)
    
    word_probit = model.out.log_prob(output_word_embedding) # (batch_size, vocab_size)
    
    # values: (batch_size, beam_size)
    values, indices = torch.topk(word_probit, beam_size) # indices: (batch_size, beam_size)

    # prepare the input data for next iteration
    current_word_input = indices.view(1, -1) # (1, batch_size*beam_size)

    previous_probit = values.view(-1, 1) # (batch_size*beam_size, 1)

    for idx in range(len(batch_idx_list)):
        for sub_idx in range(beam_size):
            previous_sequence[idx][sub_idx].append(indices[idx][sub_idx].item())

    #second iteration: construct the hidden_state for current iteration
    hidden_state = (hidden_state[0].repeat(1, 1, beam_size).view(2, -1, conf.hidden_size),  # (2, batch_size*beam_size, hidden_size)
        hidden_state[1].repeat(1, 1, beam_size).view(2, -1, conf.hidden_size)) # (2, batch_size*beam_size, hidden_size)

    for _ in range(conf.sequence_length):
        # hidden_state: ((2, batch_size, hidden_size), (2, batch_size, hidden_size))
        output_word_embedding, tmp_hidden_state = model(current_word_input, hidden_state) # review_output_embedding: (batch_size*beam_size, word_dimension)
        
        word_probit = model.out.log_prob(output_word_embedding) # (batch_size*beam_size, vocab_size)
        
        current_probit = word_probit + previous_probit # (batch_size*beam_size, vocab_size)
        
        first_values, first_indices = torch.topk(current_probit, beam_size) # (batch_size*beam_size, beam_size)

        first_values = first_values.view(batch_size, -1) # (batch_size, beam_size*beam_size)
        first_indices = first_indices.view(batch_size, -1) # (batch_size, beam_size*beam_size)

        # second_values: (batch_size, beam_size)
        second_values, second_indices = torch.topk(first_values, beam_size) # second_indices: (batch_size, beam_size)

        # collect the predicted words and store them to the dictionary
        for outer_idx, batch_topk_words_indices in enumerate(second_indices):
            for inner_idx, top_word_index in enumerate(batch_topk_words_indices):
                current_word_input[0][outer_idx*second_indices.shape[1] + inner_idx] = first_indices[outer_idx][top_word_index]
                previous_probit[outer_idx*second_indices.shape[1] + inner_idx] = first_values[outer_idx][top_word_index]
                current_sequence[outer_idx][inner_idx] = previous_sequence[outer_idx][inner_idx % 4]
                current_sequence[outer_idx][inner_idx].append(first_indices[outer_idx][top_word_index].item())
                for idx, _ in enumerate(hidden_state[0]):
                    hidden_state[0][idx][outer_idx*second_indices.shape[1] + inner_idx] = \
                        tmp_hidden_state[0][idx][outer_idx*second_indices.shape[1] + int(top_word_index.item() / 4)]
                    hidden_state[1][idx][outer_idx*second_indices.shape[1] + inner_idx] = \
                        tmp_hidden_state[1][idx][outer_idx*second_indices.shape[1] + int(top_word_index.item() / 4)]
    print(current_sequence)