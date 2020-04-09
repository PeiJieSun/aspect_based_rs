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
        self.rnn = nn.LSTM(conf.word_dimension + int(conf.embedding_dim / 2), conf.hidden_size, num_layers=1, dropout=0.4)
        self.initial_layer = nn.Linear(2*conf.embedding_dim, conf.hidden_size)

        # LOSS FUNCTIONS
        self.mse_loss = nn.MSELoss()
        self.softmax_loss = nn.AdaptiveLogSoftmaxWithLoss(\
            conf.hidden_size, conf.vocab_sz, cutoffs=[round(conf.vocab_sz/15), 3*round(conf.vocab_sz/15)], div_value=2)
        
    def forward(self, user, item, label, review_input, review_output):
        ########################### FIRST: PREDICTING RATINGS ###########################
        user_embed, item_embed = self.user_embedding(user), self.item_embedding(item) # (batch_size, 200)
        
        # rating prediction with mlp: 400 -> 200 -> 100 -> rating
        z_1 = torch.cat([user_embed, item_embed], 1)
        z_2 = F.tanh(self.mlp_1(z_1))
        z_3 = F.tanh(self.mlp_2(z_2))

        prediction = self.prediction_layer(z_3)

        rating_loss = self.mse_loss(prediction, label)

        ########################### SECOND: GENERATING REVIEWS ###########################
        h_0 = self.initial_layer(z_1).view(1, -1, conf.hidden_size)
        c_0 = self.initial_layer(z_1).view(1, -1, conf.hidden_size)
        hidden_state = (h_0, c_0)

        review_input_embed = self.word_embedding(review_input) #size: (sequence_length * batch_size * self.conf.text_word_dimension)
        #import pdb; pdb.set_trace()
        lstm_input = torch.cat([review_input_embed, z_3.repeat(review_input_embed.shape[0], 1, 1)], 2)

        outputs, hidden_state = self.rnn(lstm_input, hidden_state) # sequence_length * batch_size * hidden_size
        review_output_embed = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_size]
        #review_output_embed = outputs

        softmax_out = self.softmax_loss(review_output_embed, review_output.view(-1))
        word_probit = torch.exp(softmax_out.output)
        
        obj_loss = rating_loss + softmax_out.loss 

        return prediction, rating_loss, softmax_out.loss, obj_loss