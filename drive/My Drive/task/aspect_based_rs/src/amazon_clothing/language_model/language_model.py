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
        self.word_embedding.weight.requires_grad = False

        self.rnn = nn.GRU(conf.word_dimension, conf.hidden_size, num_layers=1)
        self.linear_5 = nn.Linear(conf.n, conf.vocab_sz)
        
    def forward(self, review_input, review_output):
        review_input_embed = self.word_embedding(review_input) #size: (sequence_length * batch_size) * self.conf.text_word_dimension
        outputs, _ = self.rnn(review_input_embed) # sequence_length * batch_size * hidden_dimension
        review_output_embed = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_dimension]
        Pwt = torch.tanh(self.linear_5(torch.cat([review_output_embed], 1)))

        obj_loss = F.nll_loss(F.log_softmax(Pwt, 1), review_output.reshape(-1), reduction='mean')
        return obj_loss