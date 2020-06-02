'''
    Description: This code is named language model, which can genreate texts based on word-level
    The input of this model is the real reviews, and each output at each time is just be influenced by the previous words

    The review generation process considers the user-item interaction information and rating information
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import string
from collections import defaultdict
from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

import config_lm_mf as conf 

class Encoder_PMF(nn.Module):
    def __init__(self):
        super(Encoder_PMF, self).__init__()
        
        torch.manual_seed(0); self.user_embedding = nn.Embedding(conf.num_users, conf.mf_dimension)
        torch.manual_seed(0); self.item_embedding = nn.Embedding(conf.num_items, conf.mf_dimension)

        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)

        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda()

        self.reinit()

    def reinit(self):
        self.user_embedding.weight = torch.nn.Parameter(0.1 * self.user_embedding.weight)
        self.item_embedding.weight = torch.nn.Parameter(0.1 * self.item_embedding.weight)
        self.user_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_users, 1))
        self.item_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_items, 1))

    def forward(self, user, item, label):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)

        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
                
        output_embed = user_embed * item_embed
                
        rating_prediction = torch.sum(output_embed, 1, keepdims=True) + self.avg_rating + user_bias + item_bias 

        obj_rating_loss = F.mse_loss(rating_prediction.view(-1), label, reduction='sum') 
        out_rating_loss = F.mse_loss(rating_prediction.view(-1), label, reduction='none')
        
        return user_embed, item_embed, rating_prediction.view(-1), obj_rating_loss, out_rating_loss

class Decoder_LM(nn.Module):
    def __init__(self):
        super(Decoder_LM, self).__init__()
        torch.manual_seed(0); self.user_linear = nn.Linear(conf.mf_dimension, conf.hidden_dimension)
        torch.manual_seed(0); self.item_linear = nn.Linear(conf.mf_dimension, conf.hidden_dimension)

        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension)
        torch.manual_seed(0); self.rnn = nn.GRU(conf.word_dimension, conf.hidden_dimension, num_layers=1, dropout=0.5)

        torch.manual_seed(0); self.rnn_out_linear = nn.Linear(conf.hidden_dimension, conf.vocab_sz)
        
    def forward(self, user_embed, item_embed, review_input, review_target):
        user_vector = self.user_linear(user_embed)
        item_vector = self.item_linear(item_embed)
        cat_vector = user_vector + item_vector

        hidden_state = torch.reshape(cat_vector, (1, -1, conf.hidden_dimension))
        #import pdb; pdb.set_trace()

        word_vector = self.word_embedding(review_input) #size: (time*batch) * self.conf.text_word_dimension
        outputs, _ = self.rnn(word_vector, hidden_state) # time * batch * hidden_dimension
        
        # convert the outputs(dimension: hidden_dimension) to dimension: vocab_sz
        outputs = self.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz) # time * batch * vocab_sz

        obj_review_loss = F.cross_entropy(outputs, review_target.reshape(-1), ignore_index=PAD, reduction='mean')
        out_review_loss = F.cross_entropy(outputs, review_target.reshape(-1), ignore_index=PAD, reduction='none')

        return obj_review_loss, out_review_loss

class Seq2Seq(nn.Module): 
    def __init__(self):
        super(Seq2Seq, self).__init__()

        encoder = Encoder_PMF()
        decoder = Decoder_LM()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, user, item, label, review_input, review_target):
        user_embed, item_embed, rating_prediction, obj_rating_loss, out_rating_loss = self.encoder(user, item, label)
        obj_review_loss, out_review_loss = self.decoder(user_embed, item_embed, review_input, review_target)
        
        obj_loss = 0.01 * obj_review_loss + 1.0 * obj_rating_loss
        return rating_prediction, out_rating_loss, out_review_loss, obj_loss
    
    def sampleTextByTemperature(self, user, item, label):
        temperature = 0.7

        user_embed, item_embed, _, _, _ = self.encoder(user, item, label)

        user_vector = self.decoder.user_linear(user_embed)
        item_vector = self.decoder.item_linear(item_embed)
        cat_vector = user_vector + item_vector

        hidden_state = torch.reshape(cat_vector, (1, 1, conf.hidden_dimension))

        sample_idx_list = [SOS]
        next_word_idx = torch.LongTensor([SOS]).cuda()
        for _ in range(conf.sequence_length):
            seed_vector = self.decoder.word_embedding(next_word_idx).reshape(1, 1, -1)

            outputs, hidden_state = self.decoder.rnn(seed_vector, hidden_state)
            word_probit = self.decoder.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)
            
            probit = word_probit.view(-1).div(temperature)
            probit = torch.clamp(probit, max=80)
            probit = probit.exp()

            next_word_idx = torch.multinomial(probit, 1)[0]
            sample_idx_list.append(next_word_idx.item())
        return sample_idx_list
    
    def sampleTextByTopOne(self, user, item, label):
        user_embed, item_embed, _, _, _ = self.encoder(user, item, label)

        user_vector = self.decoder.user_linear(user_embed)
        item_vector = self.decoder.item_linear(item_embed)
        cat_vector = user_vector + item_vector

        hidden_state = torch.reshape(cat_vector, (1, 1, conf.hidden_dimension))

        sample_idx_list = [SOS]
        next_word_idx = torch.LongTensor([SOS]).cuda()
        for _ in range(conf.sequence_length):
            seed_vector = self.decoder.word_embedding(next_word_idx).reshape(1, 1, -1)

            outputs, hidden_state = self.decoder.rnn(seed_vector, hidden_state)
            word_probit = self.decoder.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)

            next_word_idx = torch.argmax(word_probit, 1)
            sample_idx_list.append(next_word_idx.item())
        return sample_idx_list

    def sampleTextByBeamSearch(self, user, item, label):
        beam_size = 4

        previous_sequence = {}
        current_sequence = {}

        # initialize the dictionary which stores the generated words token idx
        for sub_idx in range(beam_size):
            previous_sequence[sub_idx] = [SOS]

        # first iteration: prepare the input text data
        next_word_idx = torch.LongTensor([SOS]).cuda()
        seed_vector = self.decoder.word_embedding(next_word_idx).reshape(1, 1, -1)

        user_embed, item_embed, _, _, _ = self.encoder(user, item, label)

        user_vector = self.decoder.user_linear(user_embed)
        item_vector = self.decoder.item_linear(item_embed)
        cat_vector = user_vector + item_vector

        hidden_state = torch.reshape(cat_vector, (1, 1, conf.hidden_dimension))
    
        outputs, hidden_state = self.decoder.rnn(seed_vector, hidden_state) # output_word_embedding: (1*batch_size, word_dimension)
        word_probit = self.decoder.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)

        # values: (batch_size, beam_size)
        values, indices = torch.topk(word_probit, beam_size) # indices: (batch_size, beam_size)

        # prepare the input data for next iteration
        next_word_idx = indices.view(1, -1) # (1, batch_size*beam_size)
        previous_probit = values.view(-1, 1) # (batch_size*beam_size, 1)

        for sub_idx in range(next_word_idx.shape[1]):
            previous_sequence[sub_idx].append(indices[0][sub_idx].item())

        #second iteration: construct the hidden_state for current iteration
        hidden_state = hidden_state.repeat(1, beam_size, 1)  # (1, beam_size, hidden_size)

        for _ in range(conf.sequence_length):
            seed_vector = self.decoder.word_embedding(next_word_idx)

            # hidden_state: ((2, batch_size, hidden_size), (2, batch_size, hidden_size))
            outputs, hidden_state = self.decoder.rnn(seed_vector, hidden_state) # review_output_embedding: (batch_size*beam_size, word_dimension)
            word_probit = self.decoder.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)

            tmp_hidden_state = deepcopy(hidden_state.data)

            current_probit = word_probit + previous_probit # (batch_size*beam_size, vocab_size)
            
            first_values, first_indices = torch.topk(current_probit, beam_size) # (batch_size*beam_size, beam_size)

            first_values = first_values.view(-1) # (batch_size, beam_size*beam_size)
            first_indices = first_indices.view(-1) # (batch_size, beam_size*beam_size)

            # second_values: (batch_size, beam_size)
            second_values, second_indices = torch.topk(first_values, beam_size) # second_indices: (batch_size, beam_size)

            # collect the predicted words and store them to the dictionary
            for outer_idx, top_word_index in enumerate(second_indices):
                previous_probit[outer_idx] = second_values[outer_idx]
                next_word_idx[0][outer_idx] = first_indices[top_word_index]
                current_sequence[outer_idx] = deepcopy(previous_sequence[int(top_word_index.item() / 4)])
                current_sequence[outer_idx].append(first_indices[top_word_index].item())
                hidden_state[0][outer_idx] = deepcopy(tmp_hidden_state[0][int(top_word_index.item() / 4)])

            previous_sequence = deepcopy(current_sequence)
        
        return current_sequence[0]