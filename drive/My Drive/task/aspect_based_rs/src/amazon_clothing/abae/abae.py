import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from gensim.models import Word2Vec

import config_abae as conf 

PAD = 0; SOS = 1; EOS = 2

margin_ranking_loss = nn.MarginRankingLoss(margin=1.0, reduction='none')
mse_loss = nn.MSELoss(reduction='sum')

class abae(nn.Module):
    def __init__(self):
        super(abae, self).__init__()

        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension) 
        self.word_embedding.weight.requires_grad = False
        
        self.transform_M = nn.Linear(conf.word_dimension, conf.word_dimension, bias=False) # weight: word_dimension * word_dimension
        self.transform_W = nn.Linear(conf.word_dimension, conf.aspect_dimension) # weight: aspect_dimension * word_diension
        self.transform_T = nn.Linear(conf.aspect_dimension, conf.word_dimension, bias=False) # weight: word_dimension * aspect_dimension

    # w: (batch_size, sequence_length)
    # y_s: (batch_size, word_dimension)
    # z_n: (batch_size * num_negative_reviews, word_dimension)
    def forward(self, w, y_s, z_n):
        e_w = self.word_embedding(w) # (batch_size, sequence_length, word_dimension)
        y_s = y_s.view(y_s.shape[0], y_s.shape[1], 1) # (batch_size, word_dimension, 1)
        
        # self.trainsofmr_M(e_w): (batch_size, sequence_length, word_dimension)
        dx = torch.matmul(self.transform_M(e_w), y_s) # (batch_size, sequence_length, 1)
        ax = F.softmax(dx, dim=1) # (batch_size, sequence_length, 1)     
        
        # e_w.view(e_w.shape[0], e_w.shape[2], -1): (batch_size, word_dimension, sequence_length)
        # torch.matmul(e_w, a): (batch_size, word_dimension, 1)
        z_s = torch.matmul(e_w.view(e_w.shape[0], e_w.shape[2], -1), ax).view(-1, conf.word_dimension) # (batch_size, word_dimension)

        # self.transform_W(z_s): (batch_size, aspect_dimension)
        p_t = F.softmax(self.transform_W(z_s), dim=1) # (batch_size, aspect_dimension)
        r_s = self.transform_T(p_t) # (batch_size, word_dimension)

        # cosine similarity betwee r_s and z_s
        c1 = (F.normalize(r_s, p=2, dim=1) * F.normalize(z_s, p=2, dim=1)).sum(-1, keepdim=True) # (batch_size, 1)
        c1 = c1.repeat(1, conf.num_negative_reviews).view(-1) # (batch_size * num_negative)

        # z_n.view(conf.batch_size, conf.num_negative_reviews, -1): (batch_size, num_negative_reviews, word_dimension)
        # r_s.view(conf.batch_size, 1, -1): (batch_size, 1, word_dimension)
        # z_n * r_s: (batch_size, num_negative_reviews, word_dimension)
        # (z_n * r_s).sum(-1): (batch_size, num_negative)
        # (z_n * r_s).sum(-1).sum(-1): (batch_size)
        c2 = (F.normalize(z_n.view(y_s.shape[0], conf.num_negative_reviews, -1), p=2, dim=2) \
             * F.normalize(r_s.view(y_s.shape[0], 1, -1), p=2, dim=2)).sum(-1).view(-1) # (batch_size * num_negative)
        
        out_loss = margin_ranking_loss(c1, c2, torch.FloatTensor([1.0]).cuda())

        #import pdb; pdb.set_trace()

        J_loss = torch.mean(out_loss)

        transform_T_weight = F.normalize(self.transform_T.weight, p=2, dim=0) # word_dimension * aspect_dimension
        U_loss = mse_loss(torch.matmul(torch.transpose(transform_T_weight, 0, 1), transform_T_weight), torch.eye(conf.aspect_dimension).cuda())
        return out_loss, conf.lr_lambda * U_loss + J_loss