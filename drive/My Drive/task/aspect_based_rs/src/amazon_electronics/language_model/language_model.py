import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import config_language_model as conf 

PAD = 0; SOS = 1; EOS = 2

class language_model(nn.Module):
    def __init__(self):
        super(language_model, self).__init__()
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension)
        self.rnn = nn.LSTM(conf.word_dimension, conf.hidden_size, num_layers=2, dropout=0.4)
        self.out = nn.AdaptiveLogSoftmaxWithLoss(\
            conf.hidden_size, conf.vocab_sz, cutoffs=[round(conf.vocab_sz/15), 3*round(conf.vocab_sz/15)], div_value=2)
        
    def forward(self, review_input, hidden_state):
        review_input_embedding = self.word_embedding(review_input) #size: (sequence_length * batch_size) * self.conf.text_word_dimension
        outputs, hidden_state = self.rnn(review_input_embedding, hidden_state) # sequence_length * batch_size * hidden_dimension
        review_output_embedding = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_dimension]
        
        return review_output_embedding, hidden_state