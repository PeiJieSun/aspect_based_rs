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
        self.mlp_2 = nn.Linear(conf.embedding_dim, conf.embedding_dim / 2)

        self.prediction_layer = nn.Linear(conf.embedding_dim / 2, 1)

        # PARAMETERS FOR LSTM
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension)
        self.rnn = nn.LSTM(conf.word_dimension, conf.hidden_size, num_layers=1, dropout=0.4)
        self.initial_layer = nn.Linear(2*conf.embedding_dim, conf.hidden_size)

        # LOSS FUNCTIONS
        self.mse_loss = nn.MSELoss()
        self.mse_loss_2 = nn.MSELoss(reduction='none')
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=1.0, reduction='none')
        self.softmax_loss = nn.AdaptiveLogSoftmaxWithLoss(\
            conf.hidden_size, conf.vocab_sz, cutoffs=[round(conf.vocab_sz/15), 3*round(conf.vocab_sz/15)], div_value=2)

        self.reset_para()

    def reset_para(self):
        nn.init.uniform_(self.user_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.item_embedding.weight, a=-0.1, b=0.1)

        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)
        
    def forward(self, review_input, review_output, user, item, label):
        ########################### FIRST: PREDICTING RATINGS ###########################
        user_embed, item_embed = self.user_embedding(user), self.item_embedding(item) # (batch_size, 200)
        
        # rating prediction with mlp: 400 -> 200 -> 100 -> rating
        z_1 = torch.cat([user_embed, item_embed], 1)
        z_2 = F.tanh(self.mlp_1(z_1))
        z_3 = F.tanh(self.mlp_2(z_2))

        prediction = self.prediction_layer(z_3)

        rating_loss = self.mse_loss(prediction, label)
        rating_out_loss = self.mse_loss_2(prediction, label)

        ########################### SECOND: GENERATING REVIEWS ###########################
        h_0 = self.initial_layer(z_1).view(1, -1, conf.hidden_size)
        c_0 = self.initial_layer(z_1).view(1, -1, conf.hidden_size)
        hidden_state = (h_0, c_0)

        review_input_embed = self.word_embedding(review_input) #size: (sequence_length * batch_size * self.conf.text_word_dimension)
        lstm_input = torch.cat([review_input_embed, z_3.repeat(review_input_embed.shape[0], 1, 1)], 1)

        outputs, hidden_state = self.rnn(lstm_input, hidden_state) # sequence_length * batch_size * hidden_size
        review_output_embed = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_size]
        #review_output_embed = outputs

        softmax_out = self.softmax_loss(review_output_embed, review_output.view(-1))
        word_probit = torch.exp(softmax_out.output)
        
        obj_loss = mse_loss 

        return prediction, mse_loss, generation_loss, obj_loss