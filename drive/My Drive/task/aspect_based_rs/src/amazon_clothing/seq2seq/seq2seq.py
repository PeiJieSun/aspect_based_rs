import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import config_seq2seq as conf 

PAD = 0; SOS = 1; EOS = 2

class seq2seq(nn.Module):
    def __init__(self):
        super(seq2seq, self).__init__()

        # PARAMETERS FOR LSTM
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension)
        self.rnn = nn.GRU(conf.word_dimension, conf.hidden_size, num_layers=1, dropout=0.4)

        # LOSS FUNCTIONS
        self.softmax_loss = nn.AdaptiveLogSoftmaxWithLoss(\
            conf.hidden_size, conf.vocab_sz, cutoffs=[round(conf.vocab_sz/15), 3*round(conf.vocab_sz/15)], div_value=2)
        
    def forward(self, review_input, review_output):
        ########################### FIRST: GET THE ASPECT-BASED REVIEW EMBEDDING ##########################
        review_input_embed = self.word_embedding(review_input) #size: (sequence_length, batch_size, word_dimension)
        h_0 = torch.zeros(1, review_input_embed.shape[1], conf.word_dimension).cuda()
        outputs, h_n = self.rnn(review_input_embed, h_0) # (sequence_length, batch_size, hidden_size)
        review_output_embed = outputs.view(-1, outputs.size()[2])# (sequence_length, batch_size, hidden_size)
        softmax_out = self.softmax_loss(review_output_embed, review_output.view(-1))
 
        obj_loss = softmax_out.loss
        
        return obj_loss