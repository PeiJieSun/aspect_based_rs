import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import config_expansion_net as conf 

PAD = 0; SOS = 1; EOS = 2

class expansion_net(nn.Module):
    def __init__(self):
        super(expansion_net, self).__init__()

        # PARAMETERS FOR LSTM
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension)
        self.rnn = nn.GRU(conf.word_dimension, conf.hidden_size, num_layers=1)

        self.gamma_user_embedding = nn.Embedding(conf.num_users, conf.m)
        self.gamma_item_embedding = nn.Embedding(conf.num_items, conf.m)

        self.beta_user_embedding = nn.Embedding(conf.num_users, conf.k)
        self.beta_item_embedding = nn.Embedding(conf.num_items, conf.k)

        self.u_linear = nn.Linear(2*conf.m, conf.n)
        self.v_linear = nn.Linear(2*conf.k, conf.n)

        self.linear_1 = nn.Linear(conf.m + conf.n, 1)
        self.linear_2 = nn.Linear(2*conf.k, conf.k)
        self.linear_3 = nn.Linear(conf.k+conf.word_dimension+conf.n, conf.k)
        self.linear_4 = nn.Linear(conf.n+conf.m, 1)

        self.linear_5 = nn.Linear(conf.n+conf.m, conf.vocab_sz)

        self.linear_6 = nn.Linear(conf.n, conf.vocab_sz)

        # LOSS FUNCTIONS
        self.softmax_loss = nn.AdaptiveLogSoftmaxWithLoss(\
            conf.n+conf.m, conf.vocab_sz, cutoffs=[round(conf.vocab_sz/15), 3*round(conf.vocab_sz/15)], div_value=2)
        
    def forward(self, user, item, label, review_input, review_output, review_aspect, \
                review_aspect_bool):
        ########################### FIRST: GET THE ASPECT-BASED REVIEW EMBEDDING ##########################
        gamma_u = self.gamma_user_embedding(user) # (batch_size, m)
        gamma_i = self.gamma_item_embedding(item) # (batch_size, m)

        beta_u = self.beta_user_embedding(user) # (batch_size, k)
        beta_i = self.beta_item_embedding(item) # (batch_size, k)

        u_vector = torch.tanh(self.u_linear(torch.cat([gamma_u, gamma_i], 1))) # (batch_size, n)
        v_vector = torch.tanh(self.v_linear(torch.cat([beta_u, beta_i], 1))) # (batch_size, n)

        h_0 = (u_vector + v_vector).view(1, user.shape[0], conf.hidden_size) # (1 * 1, batch_size, hidden_size=n)

        review_input_embed = self.word_embedding(review_input)# (seq_length, batch_size, word_dimension)

        outputs, h_n = self.rnn(review_input_embed) # (seq_length, batch_size, hidden_size=n)
        review_output_embed = outputs.view(-1, outputs.size()[2])#(seq_length * batch_size, hidden_size=n)
        
        # calculate a2t
        # gamma_u.repeat(outputs.shape[0], 1): (seq_length*batch_size, m)
        # torch.cat([gamma_u, review_output_embed], 1): (seq_length*batch_size, m+n)
        alpha_tu = torch.tanh(self.linear_1(torch.cat([gamma_u.repeat(outputs.shape[0], 1), review_output_embed], 1))) # (seq_length * batch_size, 1)
        
        alpha_ti = torch.tanh(self.linear_1(torch.cat([gamma_i.repeat(outputs.shape[0], 1), review_output_embed], 1))) # (seq_length * batch_size, 1)

        alpha = torch.cat([alpha_tu, alpha_ti], 1) # (seq_length * batch_size, 2)
        alpha = F.softmax(alpha, 1) # (seq_length * batch_size, 2)

        alpha = torch.transpose(alpha, 0, 1) # (2, seq_length * batch_size)
        alpha_tu, alpha_ti = alpha[0].view(-1, 1), alpha[1].view(-1, 1) # (seq_length * batch_size, 1)

        # gamma_u.view(1, user.shape[0], -1): (1, batch_size, m)
        # gamma_i.view(1, user.shape[0], -1): (1, batch_size, m)
        a2t = alpha_tu * gamma_u.repeat(outputs.shape[0], 1) + alpha_ti * gamma_i.repeat(outputs.shape[0], 1) # (seq_length * batch_size, m)
        #import pdb; pdb.set_trace()

        # calculate a3t
        # torch.cat([beta_u, beta_i], 1): (batch_size, 2*k)
        sui = self.linear_2(torch.cat([beta_u, beta_i], 1)) # (batch_size, k)

        # sui.repeat(outputs.shape[0]): (seq_length*batch_size, k)
        # torch.cat([sui.repeat(outputs.shape[0]), review_input_embed, review_output_embed], 1): (seq_length*batch_size, k+word_dim+n)
        a3t = torch.tanh(self.linear_3(torch.cat((sui.repeat(outputs.shape[0], 1), review_input_embed.view(-1, conf.word_dimension), review_output_embed), 1))) # (seq_length*batch_size, k)

        ############################### Pv(Wt) #########################################
        PvWt = torch.tanh(self.linear_5(torch.cat([review_output_embed, a2t], 1))) # (seq_length*batch_size, vocab_sz)

        ############################### P(Wt) #########################################
        #aspect_probit = torch.index_select(a3t, 1, review_aspect) * review_aspect_bool # (seq_length*batch_size, vocab_sz)
        #aspect_probit = F.log_softmax(aspect_probit, 1)

        #PvWt = torch.tanh(self.linear_6(review_output_embed))
        Pwt = PvWt# + aspect_probit
        obj_loss = F.nll_loss(F.log_softmax(Pwt, 1), review_output.view(-1), reduction='mean')

        return obj_loss