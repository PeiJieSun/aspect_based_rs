import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from gensim.models import Word2Vec

import config_abae_rs as conf

PAD = 0; SOS = 1; EOS = 2

margin_ranking_loss = nn.MarginRankingLoss(margin=1.0, reduction='mean')
mse_loss = nn.MSELoss(reduction='sum')

class encoder_abae(nn.Module):
    def __init__(self):
        super(encoder_abae, self).__init__()

        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        self.word_embedding.weight.requires_grad = False
        
        self.transform_M = nn.Linear(conf.word_dim, conf.word_dim, bias=False) # weight: word_dimension * word_dimension
        self.transform_W = nn.Linear(conf.word_dim, conf.asp_dim) # weight: aspect_dimension * word_diension
        self.transform_T = nn.Linear(conf.asp_dim, conf.word_dim, bias=False) # weight: word_dimension * aspect_dimension

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.transform_W.bias)

        torch.manual_seed(0); nn.init.xavier_uniform_(self.transform_M.weight)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.transform_W.weight)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.transform_T.weight)

    # e_w: (batch_size, sequence_length, word_dimension)
    # Y_s: (batch_size, word_dimension)
    # pos_rev: (batch_size, sequence_length)
    def _attention(self, pos_rev, e_w, y_s):
        mask = (pos_rev > 0).long() # (batch_size, seq_len)

        # self.transform_M(e_w): (batch_size, sequence_length, word_dimension)
        dx = torch.matmul(self.transform_M(e_w), y_s).view(-1, conf.seq_len) # (batch_size, sequence_length)
        dx_mask = (dx < 80).long()
        dx = dx_mask * dx

        ax_1 = torch.exp(dx) # (batch_size, seq_len)
        ax_2 = ax_1 * mask # (batch_size, seq_len)
        ax_3 = torch.sum(ax_2, dim=1, keepdim=True) + 1e-6 # (batch_size, 1)
        ax_4 = (ax_2 / ax_3).view(-1, conf.seq_len, 1) # (batch_size, seq_len, 1)
        
        # e_w.transpose(1, 2): (batch_size, word_dimension, sequence_length)
        # torch.matmul(e_w, a): (batch_size, word_dimension, 1)
        z_s = torch.sum(e_w*ax_4, dim=1).view(-1, conf.word_dim) # (batch_size, word_dimension)

        return z_s

    # w: (batch_size, sequence_length)
    # y_s: (batch_size, word_dimension)
    # z_n: (batch_size * num_negative_reviews, word_dimension)
    def forward(self, pos_rev, neg_rev):
        #positive_review: (batch_size, sequence_length)
        #negative_review: (batch_size*num_negative_reviews, sequence_length)

        pos_rev_emb = self.word_embedding(pos_rev) # (batch_size, sequence_length, word_dimension)
        neg_rev_emb = self.word_embedding(neg_rev) # (batch_size*num_negative_reviews, sequence_length, word_dimension)

        y_s = torch.sum(pos_rev_emb, 1) # (batch_size, word_dimension, 1)
        z_n = torch.sum(neg_rev_emb, 1) # (batch_size * num_negative_reviews, word_dimension)
        
        pos_rev_mask = (pos_rev > 0).long()
        neg_rev_mask = (neg_rev > 0).long()

        pos_rev_mask = torch.sum(pos_rev_mask, dim=1, keepdim=True) + 1e-6
        neg_rev_mask = torch.sum(neg_rev_mask, dim=1, keepdim=True) + 1e-6

        y_s = (y_s / pos_rev_mask).view(-1, conf.word_dim, 1)
        z_n = (z_n / neg_rev_mask).view(-1, conf.word_dim)

        z_s = self._attention(pos_rev, pos_rev_emb, y_s) # (batch_size, word_dimension)
        
        #p_t = self.transform_W(z_s)
        p_t = F.softmax(self.transform_W(z_s), dim=1) # (batch_size, aspect_dimension)
        r_s = self.transform_T(p_t) # (batch_size, word_dimension)

        # cosine similarity betwee r_s and z_s
        c1 = (F.normalize(r_s, p=2, dim=1) * F.normalize(z_s, p=2, dim=1)).sum(-1, keepdim=True) # (batch_size, 1)
        c1 = c1.repeat(1, conf.num_neg_sent).view(-1) # (batch_size * num_negative)

        # z_n.view(conf.batch_size, conf.num_negative_reviews, -1): (batch_size, num_negative_reviews, word_dimension)
        # r_s.view(conf.batch_size, 1, -1): (batch_size, 1, word_dimension)
        # z_n * r_s: (batch_size, num_negative_reviews, word_dimension)
        # (z_n * r_s).sum(-1): (batch_size, num_negative)
        # (z_n * r_s).sum(-1).view(-1): (batch_size)
        c2 = (F.normalize(z_n.view(y_s.shape[0], conf.num_neg_sent, -1), p=2, dim=2) \
             * F.normalize(r_s.view(y_s.shape[0], 1, -1), p=2, dim=2)).sum(-1).view(-1) # (batch_size * num_negative)
        
        J_loss = margin_ranking_loss(c1, c2, torch.FloatTensor([1.0]).cuda())
    
        transform_T_weight = F.normalize(self.transform_T.weight, p=2, dim=0) # word_dimension * asp_dim
        U_loss = mse_loss(torch.matmul(torch.transpose(transform_T_weight, 0, 1), transform_T_weight), torch.eye(conf.asp_dim).cuda())

        return p_t, J_loss, U_loss

class decoder_fm(nn.Module):
    def __init__(self):
        super(decoder_fm, self).__init__()
        
        self.user_fc_linear = nn.Linear(conf.asp_dim, conf.mf_dim)
        self.item_fc_linear = nn.Linear(conf.asp_dim, conf.mf_dim)
        
        torch.manual_seed(0); \
            self.gmf_user_embedding = nn.Embedding(conf.num_users, conf.gmf_embed_dim)
        torch.manual_seed(0); \
            self.gmf_item_embedding = nn.Embedding(conf.num_items, conf.gmf_embed_dim)

        self.linears = []
        for idx in range(1, len(conf.mlp_dim_list)):
            self.linears.append(nn.Linear(conf.mlp_dim_list[idx-1], conf.mlp_dim_list[idx], bias=False).cuda())

        self.x_linears = []
        for idx in range(1, len(conf.mlp_dim_list)):
            self.x_linears.append(nn.Linear(conf.mlp_dim_list[idx-1], conf.mlp_dim_list[idx], bias=False).cuda())

        self.final_linear = nn.Linear(conf.mlp_embed_dim+conf.gmf_embed_dim, 1)
        self.x_final_linear = nn.Linear(conf.gmf_embed_dim, 1)

        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)

        self.reinit()

    def reinit(self):
        for fc in [self.user_fc_linear, self.item_fc_linear]:
            torch.manual_seed(0); nn.init.uniform_(fc.weight, -0.1, 0.1)
            torch.manual_seed(0); nn.init.constant_(fc.bias, 0.1)

        self.gmf_user_embedding.weight = \
            torch.nn.Parameter(0.1 * self.gmf_user_embedding.weight)
        self.gmf_item_embedding.weight = \
            torch.nn.Parameter(0.1 * self.gmf_item_embedding.weight)

        for idx in range(len(conf.mlp_dim_list)-1):
            torch.manual_seed(0); nn.init.uniform_(self.linears[idx].weight, -0.1, 0.1)

        for idx in range(len(conf.mlp_dim_list)-1):
            torch.manual_seed(0); nn.init.uniform_(self.x_linears[idx].weight, -0.1, 0.1)
    
        torch.manual_seed(0); nn.init.uniform_(self.final_linear.weight, -0.05, 0.05)
        nn.init.zeros_(self.final_linear.bias)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        torch.manual_seed(0); nn.init.uniform_(self.x_final_linear.weight, -0.05, 0.05)
        torch.manual_seed(0); nn.init.constant_(self.x_final_linear.bias, 0.0)


    def forward(self, user, item, aspect_user_embed, aspect_item_embed):
        gmf_user_embed = self.gmf_user_embedding(user)
        gmf_item_embed = self.gmf_item_embedding(item)
        
        gmf_concat_embed = torch.cat([gmf_user_embed, gmf_item_embed], dim=1)
        for idx in range(len(conf.mlp_dim_list)-1):
            gmf_concat_embed = torch.relu(self.x_linears[idx](gmf_concat_embed))

        mlp_user_embed = self.user_fc_linear(aspect_user_embed)
        mlp_item_embed = self.item_fc_linear(aspect_item_embed)

        mlp_concat_emebd = torch.cat([mlp_user_embed, mlp_item_embed], dim=1)
        for idx in range(len(conf.mlp_dim_list)-1):
            mlp_concat_emebd = torch.relu(self.linears[idx](mlp_concat_emebd))
        
        final_embed = torch.cat([gmf_concat_embed, mlp_concat_emebd], dim=1)
        
        #import pdb; pdb.set_trace()

        prediction = self.x_final_linear(1.0*mlp_concat_emebd + 1.0*gmf_concat_embed) + conf.avg_rating + \
            self.user_bias(user) + self.item_bias(item)

        return prediction.view(-1)

class abae_rs(nn.Module):
    def __init__(self):
        super(abae_rs, self).__init__()

        self.encoder = encoder_abae()
        self.decoder = decoder_fm()

        '''1_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### FIRST PART #### 
        '''
        torch.manual_seed(0); self.embedding_user = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.embedding_item = nn.Embedding(conf.num_items, conf.mf_dim)
        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)
        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda() 
        
        self.reinit() ### '''
        #### ****** veriify rating prediction with PMF ****** ------ END ####

        


    def reinit(self):
        '''2_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### SECOND PART #### 
        '''
        self.embedding_user.weight = torch.nn.Parameter(0.1 * self.embedding_user.weight)
        self.embedding_item.weight = torch.nn.Parameter(0.1 * self.embedding_item.weight)
        self.user_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_users, 1))
        self.item_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_items, 1)) ### '''
        #### ****** veriify rating prediction with PMF ****** ------ END ####

        

    def forward(self, user, item, label, user_pos_sent, user_neg_sent, item_pos_sent, item_neg_sent):
        aspect_user_embed, user_J_loss, user_U_loss = self.encoder(user_pos_sent, user_neg_sent)
        aspect_item_embed, item_J_loss, item_U_loss = self.encoder(item_pos_sent, item_neg_sent)

        aspect_user_embed = torch.mean(aspect_user_embed.reshape(-1, conf.user_seq_num, conf.asp_dim), dim=1)
        aspect_item_embed = torch.mean(aspect_item_embed.reshape(-1, conf.item_seq_num, conf.asp_dim), dim=1)

        pred = self.decoder(user, item, aspect_user_embed, aspect_item_embed)

        '''3_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### THIRD PART #### 
        '''
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        output_emb = user_emb * item_emb
        x_prediction = torch.sum(output_emb, 1, keepdims=True) + self.avg_rating + user_bias + item_bias
        pred = x_prediction.view(-1)  ### '''
        #### START ------ ****** veriify rating prediction with PMF ****** ####

        rating_out_loss = F.mse_loss(pred, label, reduction='none')
        rating_obj_loss = F.mse_loss(pred, label, reduction='sum')

        obj_loss = 1.0*rating_obj_loss + 1e-8*(user_J_loss+item_J_loss+user_U_loss+item_U_loss)
        return pred, obj_loss, rating_out_loss