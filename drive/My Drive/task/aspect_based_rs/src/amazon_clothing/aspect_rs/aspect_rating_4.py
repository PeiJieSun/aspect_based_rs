# Co-attention https://arxiv.org/pdf/1606.00061.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_aspect_4 as conf

class aspect_rating_4(nn.Module):
    def __init__(self):
        super(aspect_rating_4, self).__init__()

        # parameters for ABAE
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension) 
        self.word_embedding.weight.requires_grad = False
        
        self.transform_M = nn.Linear(conf.word_dimension, conf.word_dimension, bias=False) # weight: word_dimension * word_dimension
        self.transform_W = nn.Linear(conf.word_dimension, conf.aspect_dimension) # weight: aspect_dimension * word_diension
        self.transform_T = nn.Linear(conf.aspect_dimension, conf.word_dimension, bias=False) # weight: word_dimension * aspect_dimension

        self.user_fc_linear = nn.Linear(conf.common_dimension, conf.embedding_dim)

        # parameters for Co-Attention
        self.Wb = nn.Parameter(torch.randn(1, conf.common_dimension, conf.common_dimension))
        self.Wq = nn.Parameter(torch.randn(1, conf.co_att_k, conf.common_dimension))
        self.Wv = nn.Parameter(torch.randn(1, conf.co_att_k, conf.common_dimension))

        self.whv = nn.Parameter(torch.randn((1, 1, conf.co_att_k)))
        self.whq = nn.Parameter(torch.randn((1, 1, conf.co_att_k)))

        # parameters for FM
        dim = conf.embedding_dim * 2
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(conf.num_users, 1))
        self.b_items = nn.Parameter(torch.randn(conf.num_items, 1))

        self.dropout = nn.Dropout(conf.drop_out)

        self.mse_func_1 = nn.MSELoss(reduction='none')
        self.mse_func_2 = nn.MSELoss()
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=1.0, reduction='none')

        self.reset_para()

    def reset_para(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def forward(self, historical_review, # (num_review, sequence_length)
        neg_review, # (num_review * num_negative, word_dimension)
        user, item, label, user_idx_list, item_idx_list): # (batch_size, 1)
        ########################### First: encode the historical review, and get the aspect-based review embedding ###########################
        # encode the historical reviews of the target user and item(mapping the historical reviews of the target user and item to the aspect space)
        # historical_review_set ==> aspect-based historical review embedding
        w = historical_review; 

        positive_review_embed = self.word_embedding(w) # (batch_size, sequence_length, word_dimension)
        negative_review_embed = self.word_embedding(neg_review) # (batch_size*num_negative_reviews, sequence_length, word_dimension)

        y_s = torch.mean(positive_review_embed, 1).view(-1, conf.word_dimension) # (batch_size, word_dimension)
        z_n = torch.mean(negative_review_embed, 1).view(-1, conf.word_dimension) # (batch_size * num_negative_reviews, word_dimension)

        e_w = self.word_embedding(w) # (num_review, sequence_length, word_dimension)
        y_s = y_s.view(y_s.shape[0], y_s.shape[1], 1) # (num_review, word_dimension, 1)
        
        # self.trainsofmr_M(e_w): (num_review, sequence_length, word_dimension)
        dx = torch.matmul(self.transform_M(e_w), y_s) # (num_review, sequence_length, 1)
        ax = F.softmax(dx, dim=1) # (num_review, sequence_length, 1)     
        
        # e_w.view(e_w.shape[0], e_w.shape[2], -1): (num_review, word_dimension, sequence_length)
        # torch.matmul(e_w, a): (num_review, word_dimension, 1)
        z_s = torch.matmul(e_w.view(e_w.shape[0], e_w.shape[2], -1), ax).view(-1, conf.word_dimension) # (num_review, word_dimension)

        # self.transform_W(z_s): (num_review, aspect_dimension)
        #p_t = F.softmax(self.transform_W(z_s), dim=1) # (num_review, aspect_dimension)
        p_t = self.transform_W(z_s)
        r_s = self.transform_T(p_t) # (num_review, word_dimension)
        
        # cosine similarity betwee r_s and z_s
        c1 = (F.normalize(r_s, p=2, dim=1) * F.normalize(z_s, p=2, dim=1)).sum(-1, keepdim=True) # (num_review, 1)
        c1 = c1.repeat(1, conf.num_negative_reviews).view(-1) # (num_review * num_negative)

        # z_n.view(conf.num_review, conf.num_negative_reviews, -1): (num_review, num_negative_reviews, word_dimension)
        # r_s.view(conf.num_review, 1, -1): (num_review, 1, word_dimension)
        # z_n * r_s: (num_review, num_negative_reviews, word_dimension)
        # (z_n * r_s).sum(-1): (num_review, num_negative)
        # (z_n * r_s).sum(-1).sum(-1): (num_review)
        c2 = (F.normalize(z_n.view(y_s.shape[0], conf.num_negative_reviews, -1), p=2, dim=2) \
             * F.normalize(r_s.view(y_s.shape[0], 1, -1), p=2, dim=2)).sum(-1).view(-1) # (num_review * num_negative)

        abae_out_loss = self.margin_ranking_loss(c1, c2, torch.FloatTensor([1.0]).cuda())

        J_loss = torch.mean(abae_out_loss)

        transform_T_weight = F.normalize(self.transform_T.weight, p=2, dim=0) # word_dimension * aspect_dimension
        U_loss = self.mse_func_2(torch.matmul(torch.transpose(transform_T_weight, 0, 1), transform_T_weight), torch.eye(conf.aspect_dimension).cuda())
        
        ########################### Second: collect the aspect-based user embedding and item embedding ###########################
        aspect_user_embed = p_t[user_idx_list] # (batch_size * u_max_r, xx_dim)
        aspect_item_embed = p_t[item_idx_list] # (batch_size * i_max_r, xx_dim)
        
        # (batch_size, xx_dim, u_max_r)
        Q = torch.transpose(aspect_user_embed.view(label.shape[0], -1, conf.common_dimension), 1, 2)
        # (batch_size, xx_dim, i_max_r)
        V = torch.transpose(aspect_item_embed.view(label.shape[0], -1, conf.common_dimension), 1, 2)
        # self.transpose(Q, 1, 2): (batch_size, u_max_r, xx_dim)
        # self.Wb.view(1, conf.common_dimension, conf.common_dimension): (batch_size, xx_dim, xx_dim)
        # torch.matmul(torch.matmul(self.transpose(Q, 1, 2), \
        #    self.Wb.view(1, conf.common_dimension, conf.common_dimension)): (batch_size, u_max_r, xx_dim)
        C = F.tanh(torch.matmul(torch.matmul(torch.transpose(Q, 1, 2), \
            self.Wb.view(1, conf.common_dimension, conf.common_dimension)), V)) # (batch_size, u_max_r, i_max_r)
                
        # self.Wv: (1, k, xx_dim)
        WvV = torch.matmul(self.Wv, V) # (batch_size, k, i_max_r)

        # self.Wq: (k, xx_dim)
        WqQ = torch.matmul(self.Wq, Q) # (batch_size, k, u_max_r)
        
        # WqQ: (batch_size, k, u_max_r)
        # C: (batch_size, u_max_r, i_max_r)
        # torch.matmul(WqQ, C): (batch_size, k, i_max_r)
        # WvV: (batch_size, k, i_max_r)
        Hv = F.tanh(WvV + torch.matmul(WqQ, C)) # (batch_size, k, i_max_r)
        # WvV: (batch_size, k, i_max_r)
        # torch.transpose(C, 1, 2): (batch_size, i_max_r, u_max_r)
        # torch.matmul(WvV, torch.transpose(C, 1, 2)): (batch_size, k, u_max_r)
        # WqQ: (batch_size, k, u_max_r)
        Hq = F.tanh(WqQ + torch.matmul(WvV, torch.transpose(C, 1, 2)))
        
        # self.whv: (1, k)
        # Hv: (batch_size, k, i_max_r)
        # torch.matmul(self.whv, Hv): (batch_size, 1, i_max_r)
        av = F.softmax(torch.matmul(self.whv, Hv), dim=2)
        # self.whq: (1, k)
        # Hq: (batch_size, k, u_max_r)
        # torch.matmul(self.whq, Hq): (batch_size, 1, u_max_r)
        aq = F.softmax(torch.matmul(self.whq, Hq), dim=2)

        _V = av * V # (batch_size, xx_dim, i_max_r)
        _Q = aq * Q # (batch_size, xx_dim, u_max_r)
        
        user_aspect_embed = torch.sum(_V, 2)
        item_aspect_embed = torch.sum(_Q, 2)

        #import pdb; pdb.set_trace()
        ########################### Third: predict ratings with FM module ###########################
        user_aspect_embed = self.user_fc_linear(user_aspect_embed)
        item_aspect_embed = self.user_fc_linear(item_aspect_embed)

        u_out = self.dropout(user_aspect_embed) #+ self.free_user_embedding(user)
        i_out = self.dropout(item_aspect_embed) #+ self.free_item_embedding(item)

        input_vec = torch.cat([u_out, i_out], 1)
        
        input_vec = self.dropout(input_vec)

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part + self.b_users[user] + self.b_items[item] # + conf.avg_rating

        prediction = fm_output.squeeze(1)
        
        rating_loss = self.mse_func_1(prediction, label)
        mse_loss = self.mse_func_2(prediction, label)

        # collect the loss of abae and rating prediction
        obj_loss = mse_loss + 0.001*J_loss + 0.001*U_loss
        
        return obj_loss, rating_loss, abae_out_loss, prediction
