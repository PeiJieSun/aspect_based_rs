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
        self.linear_5 = nn.Linear(conf.hidden_size, conf.vocab_sz)

        self.user_embedding = nn.Embedding(conf.num_users, conf.mf_dimension)
        self.item_embedding = nn.Embedding(conf.num_items, conf.mf_dimension)

        self.linear_1 = nn.Linear(2*conf.mf_dimension, conf.hidden_size)
        
    def forward(self, user, item, review_input, review_output):
        
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)

        hidden_state = \
            self.linear_1(torch.cat([user_embed, item_embed], 1)).view(1, -1, conf.hidden_size) #(1, batch_size, hidden_size)

        current_input_embed = self.word_embedding(review_input[0]).view(1, -1, conf.word_dimension) #(1, batch_size, word_dimension)

        Pwt = []
        for idx, _ in enumerate(review_input):
            outputs, _ = self.rnn(current_input_embed, hidden_state) # sequence_length * batch_size * hidden_dimension
            review_output_embed = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_dimension]
            tmp_Pwt = self.linear_5(torch.cat([review_output_embed], 1))
            Pwt.append(tmp_Pwt)

            tmp_word = torch.argmax(tmp_Pwt, 1)
            #import pdb; pdb.set_trace()
            print(tmp_word)
            current_input_embed = self.word_embedding(tmp_word).view(1, -1, conf.word_dimension)
        
        #import pdb; pdb.set_trace()
        obj_loss = F.nll_loss(F.log_softmax(torch.cat(Pwt, dim=0), 1), review_output.reshape(-1), reduction='mean')
        return obj_loss
        
    def generate(self, hidden_state, review_input):
        review_input_embed = self.word_embedding(review_input) #size: (sequence_length * batch_size) * self.conf.text_word_dimension
        outputs, hidden_state = self.rnn(review_input_embed, hidden_state) # sequence_length * batch_size * hidden_dimension
        review_output_embed = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_dimension]
        Pwt = torch.tanh(self.linear_5(torch.cat([review_output_embed], 1)))

        return F.log_softmax(Pwt, 1), hidden_state