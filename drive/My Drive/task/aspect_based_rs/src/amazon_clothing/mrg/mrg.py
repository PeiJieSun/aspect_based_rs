import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import config_mrg as conf 

PAD = 0; SOS = 1; EOS = 2

class mrg(nn.Module):
    def __init__(self):
        super(mrg, self).__init__()
        
        # PARAMETERS FOR RATING PREDICTION
        self.user_embedding = nn.Embedding(conf.num_users, conf.embedding_dim)
        self.item_embedding = nn.Embedding(conf.num_items, conf.embedding_dim)

        self.mlp_1 = nn.Linear(2*conf.embedding_dim, conf.embedding_dim)
        self.mlp_2 = nn.Linear(conf.embedding_dim, int(conf.embedding_dim / 2))

        self.prediction_layer = nn.Linear(int(conf.embedding_dim / 2), 1)

        # PARAMETERS FOR LSTM
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension)
        #self.rnn = nn.GRU(conf.word_dimension + int(conf.embedding_dim / 2), conf.hidden_size, num_layers=1, dropout=0.4)
        self.rnn = nn.GRU(conf.word_dimension, conf.hidden_size, num_layers=1, dropout=0.4)
        self.initial_layer = nn.Linear(2*conf.embedding_dim, conf.hidden_size)

        self.linear = nn.Linear(conf.word_dimension, conf.vocab_sz)

        # LOSS FUNCTIONS
        self.mse_loss = nn.MSELoss()
        
    def forward(self, user, item, label, review_input, review_output):
        ########################### FIRST: PREDICTING RATINGS ###########################
        user_embed, item_embed = self.user_embedding(user), self.item_embedding(item) # (batch_size, 200)
        
        # rating prediction with mlp: 400 -> 200 -> 100 -> rating
        z_1 = torch.cat([user_embed, item_embed], 1) # (batch_size, 2*embedding_dim)
        z_2 = F.tanh(self.mlp_1(z_1)) # (batch_size, embedding_dim)
        z_3 = F.tanh(self.mlp_2(z_2)) # (batch_size, embedding_dim)

        #prediction = self.prediction_layer(z_3)
        #rating_loss = self.mse_loss(prediction, label)
        
        ########################### SECOND: GENERATING REVIEWS ###########################
        h_0 = self.initial_layer(z_1).view(1, -1, conf.hidden_size)  #(1, batch_size, hidden_size)

        review_input_embed = self.word_embedding(review_input) #size: (sequence_length * batch_size * self.conf.text_word_dimension)
        #import pdb; pdb.set_trace()
        #lstm_input = torch.cat([review_input_embed, z_3.repeat(review_input_embed.shape[0], 1, 1)], 2)
        lstm_input = review_input_embed

        outputs, h_n = self.rnn(lstm_input, h_0) # sequence_length * batch_size * hidden_size
        review_output_embed = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_size]

        Pwt = torch.tanh(self.linear(review_output_embed))

        obj_loss = F.nll_loss(F.log_softmax(Pwt, 1), review_output.view(-1), reduction='mean')

        return obj_loss