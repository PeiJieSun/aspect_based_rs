import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from gensim.models import Word2Vec

import config_abae as conf

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

        #import  pdb; pdb.set_trace()
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

        #import pdb; pdb.set_trace()

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
        
        out_loss = margin_ranking_loss(c1, c2, torch.FloatTensor([1.0]).cuda())
        
        J_loss = torch.mean(out_loss)

        transform_T_weight = F.normalize(self.transform_T.weight, p=2, dim=0) # word_dimension * aspect_dimension
        U_loss = mse_loss(torch.matmul(torch.transpose(transform_T_weight, 0, 1), transform_T_weight), torch.eye(conf.asp_dim).cuda())
        return out_loss, U_loss, J_loss

class decoder_fm(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.free_user_embedding = nn.Embedding(conf.num_users, conf.embedding_dim)  # user/item num * 32
        self.free_item_embedding = nn.Embedding(conf.num_items, conf.embedding_dim)

        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Embedding(conf.num_users, 1)
        self.b_items = nn.Embedding(conf.num_items, 1)

        self.reinit()

    def reinit(self):
        nn.init.xavier_normal_(self.free_user_embedding.weight)
        nn.init.xavier_normal_(self.free_item_embedding.weight)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.b_users.bias)
        nn.init.zeros_(self.b_items.bias)
        nn.init.xavier_normal_(self.fm_V)

    def forward(self, user, item, label, aspect_user_embed, aspect_item_embed):
        item_aspect_embed = self.transform_wang(item_aspect_embed)

        # Co-Attention
        aspect_user_embed = torch.sum(torch.matmul(aspect_user_embed.view(label.shape[0], -1, 1), \
            aspect_item_embed.view(label.shape[0], 1, -1)), 1)

        final_user_embed = aspect_user_embed + self.free_user_embedding(user)
        final_item_embed = aspect_item_embed + self.free_item_embedding(item)

        input_vec = torch.cat([final_user_embed, final_item_embed], 1) # (batch_size, 2*xx_dimension)
        input_vec = self.user_fc_linear(input_vec)
        input_vec = self.dropout(input_vec)

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part + self.b_users[user] + self.b_items[item] # + conf.avg_rating

        prediction = fm_output.squeeze(1)
        
        return prediction


class abae_rs(nn.Module):
    def __init__(self):
        super(abae_rs, self).__init__()

        self.encoder = encoder_abae()
        self.decoder = decoder_fm()

    def forward(self, pos_user_rev, neg_user_rev, pos_item_rev, neg_item_rev, user, item, label):
        aspect_user_embed = self.encoder_abae(pos_user_rev, neg_user_rev)
        aspect_item_embed = self.encoder_abae(pos_item_rev, neg_item_rev)

        aspect_user_embed = torch.mean(aspect_user_embed.reshape(-1, conf.user_seq_num, conf.word_dim))
        aspect_item_embed = torch.mean(aspect_item_embed.reshape(-1, conf.item_seq_num, conf.word_dim))

        rating_out_loss = F.mse_loss(prediction, label, reduction='sum')
        rating_obj_loss = F.mse_loss(prediction, label, reduction='none')

        return rating_out_loss, rating_obj_loss