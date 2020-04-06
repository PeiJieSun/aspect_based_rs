import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import config_generation as conf 

PAD = 0; SOS = 1; EOS = 2

class generation(nn.Module):
    def __init__(self):
        super(generation, self).__init__()

        # PARAMETERS FOR LSTM
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension)
        self.rnn = nn.LSTM(conf.word_dimension, conf.hidden_size, num_layers=1, dropout=0.4)
        self.init_tranfer_layer = nn.Linear(2*conf.common_dimension, conf.hidden_size)
        
        # PARAMETERS FOR ASPECT EXTRACTION
        self.transform_M = nn.Linear(conf.word_dimension, conf.word_dimension, bias=False) # weight: word_dimension * word_dimension
        self.transform_W = nn.Linear(conf.word_dimension, conf.aspect_dimension) # weight: aspect_dimension * word_diension
        self.transform_T = nn.Linear(conf.aspect_dimension, conf.word_dimension, bias=False) # weight: word_dimension * aspect_dimension

        self.free_user_embedding = nn.Embedding(conf.num_users, conf.common_dimension)
        self.free_item_embedding = nn.Embedding(conf.num_items, conf.common_dimension)

        # PARAMETERS FOR FM
        self.aspect_user_embedding = nn.Embedding(conf.num_users, conf.common_dimension)  # user/item num * 32
        self.aspect_item_embedding = nn.Embedding(conf.num_items, conf.common_dimension)
        self.aspect_user_embedding.weight.requires_grad = False
        self.aspect_item_embedding.weight.requires_grad = False

        dim = conf.common_dimension * 2
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(conf.num_users, 1))
        self.b_items = nn.Parameter(torch.randn(conf.num_items, 1))

        self.dropout = nn.Dropout(conf.drop_out)

        '''
            DEBUG
        '''
        self.user_embedding = nn.Embedding(conf.num_users, conf.embedding_dim)  # user/item num * 32
        self.item_embedding = nn.Embedding(conf.num_items, conf.embedding_dim)

        # LOSS FUNCTIONS
        self.mse_loss = nn.MSELoss()
        self.mse_loss_2 = nn.MSELoss(reduction='none')
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=1.0, reduction='none')
        self.softmax_loss = nn.AdaptiveLogSoftmaxWithLoss(\
            conf.hidden_size, conf.vocab_sz, cutoffs=[round(conf.vocab_sz/15), 3*round(conf.vocab_sz/15)], div_value=2)

        self.reset_para()

    def reset_para(self):
        nn.init.uniform_(self.free_user_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.free_item_embedding.weight, a=-0.1, b=0.1)

        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)
        
    def forward(self, review_input, review_output, review_aspect, review_aspect_bool,
        historical_review, # (num_review, sequence_length)
        neg_review, # (num_review * num_negative, word_dimension)
        user, item, label, # (batch_size, 1)
        user_histor_index, user_histor_value, item_histor_index, item_histor_value):
        ########################### FIRST: GET THE ASPECT-BASED REVIEW EMBEDDING ###########################
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
        U_loss = self.mse_loss(torch.matmul(torch.transpose(transform_T_weight, 0, 1), transform_T_weight), torch.eye(conf.aspect_dimension).cuda())
        
        ########################### SECOND: CLLECT THE ASPECT-BASED USER & ITEM EMBEDDING ###########################
        # get the embedding of the user and the item
        user_histor_tensor = \
            torch.sparse.FloatTensor(user_histor_index, user_histor_value, torch.Size([label.shape[0], w.shape[0]])) # (batch_size, num_review)
        item_histor_tensor = \
            torch.sparse.FloatTensor(item_histor_index, item_histor_value, torch.Size([label.shape[0], w.shape[0]])) # (batch_size, num_review)

        '''
        user_histor_tensor = torch.sparse.FloatTensor(user_histor_index, user_histor_value, \
            torch.Size([label.shape[0], w.shape[0]])).to_dense() # (batch_size, num_review)
        item_histor_tensor = torch.sparse.FloatTensor(item_histor_index, item_histor_value, \
            torch.Size([label.shape[0], w.shape[0]])).to_dense() # (batch_size, num_review)
        '''

        ########################### THIRD: PREDICT THE RATINNG OF USER-ITEM ###########################
        aspect_user_embed = torch.mm(user_histor_tensor, r_s) # (batch_size, word_dimension)
        aspect_item_embed = torch.mm(item_histor_tensor, r_s) # (batch_size, word_dimension)
        
        u_out = aspect_user_embed # (batch_size, word_dimension)
        i_out = aspect_item_embed # (batch_size, word_dimension)

        free_user_embed = self.free_user_embedding(user) # (batch_size, common_dimension)
        free_item_embed = self.free_item_embedding(item) # (batch_size, common_dimension)

        u_out = self.dropout(u_out.reshape(u_out.size(0), -1)) + free_user_embed # (batch_size, common_dimension)
        i_out = self.dropout(i_out.reshape(i_out.size(0), -1)) + free_item_embed # (batch_size, common_dimension)

        input_vec = torch.cat([u_out, i_out], 1) # (batch_size, 2*common_dimension)

        fm_linear_part = self.fc(input_vec) # (batch_size, 1)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V) # (batch_size, 10)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2) # (batch_size, 10)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2), # (batch_size, 10)
                                     torch.pow(self.fm_V, 2)) # (batch_size, 10)
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdims=True) + fm_linear_part + self.b_users[user] + self.b_items[item] # + conf.avg_rating

        prediction = fm_output.squeeze(1)
        
        mse_loss = self.mse_loss(prediction, label)

        ########################### FOURTH: GENERATE REVIEWS ###########################
        h_0 = self.init_tranfer_layer(input_vec).view(1, -1, conf.hidden_size)
        c_0 = self.init_tranfer_layer(input_vec).view(1, -1, conf.hidden_size)
        hidden_state = (h_0, c_0)

        review_input_embed = self.word_embedding(review_input) #size: (sequence_length * batch_size) * self.conf.text_word_dimension
        outputs, hidden_state = self.rnn(review_input_embed, hidden_state) # sequence_length * batch_size * hidden_size
        review_output_embed = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_size]
        #review_output_embed = outputs

        softmax_out = self.softmax_loss(review_output_embed, review_output.view(-1))
        word_probit = torch.exp(softmax_out.output)
        #import pdb; pdb.set_trace()

        # Calculate the aspect loss
        aspect_user_sentiment = torch.mm(user_histor_tensor, p_t) # (batch_size, word_dimension)
        aspect_item_sentiment = torch.mm(item_histor_tensor, p_t) # (batch_size, word_dimension)
        aspect_sentiment = aspect_user_sentiment * aspect_item_sentiment

        review_aspect_embed = F.one_hot(review_aspect, num_classes=conf.aspect_dimension)
        aspect_sentiment = aspect_sentiment.view(-1, 1, 1, conf.aspect_dimension)
        review_aspect_bool = review_aspect_bool.view(user.shape[0], -1, 1, 1)

        #import pdb; pdb.set_trace()
        aspect_probit = torch.sum(review_aspect_embed * aspect_sentiment * review_aspect_bool, -1, keepdim=True)
        #import pdb; pdb.set_trace()

        generation_loss = torch.mean(-torch.log(word_probit + aspect_probit.view(-1)))
        
        obj_loss = mse_loss + J_loss + U_loss + generation_loss

        #import pdb; pdb.set_trace()
        return aspect_user_embed, aspect_item_embed, prediction, mse_loss, J_loss, generation_loss, obj_loss

    def predict(self, user, item, label):
        '''
        u_fea = self.aspect_user_embedding(user)
        i_fea = self.aspect_item_embedding(item)

        #import pdb; pdb.set_trace()
        u_out = u_fea.view(-1, 1, conf.common_dimension)
        i_out = i_fea.view(-1, 1, conf.common_dimension)
        '''

        u_fea = self.user_embedding(user)
        i_fea = self.item_embedding(item)

        u_out = u_fea.view(-1, 1, conf.embedding_dim)
        i_out = i_fea.view(-1, 1, conf.embedding_dim)

        free_user_embed = self.free_user_embedding(user)
        free_item_embed = self.free_item_embedding(item)

        u_out = self.dropout(u_out.reshape(u_out.size(0), -1)) #+ free_user_embed
        i_out = self.dropout(i_out.reshape(i_out.size(0), -1)) #+ free_item_embed

        input_vec = torch.cat([u_out, i_out], 1)

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2), torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdims=True) + fm_linear_part #+ self.b_users[user] + self.b_items[item] # + conf.avg_rating

        prediction = fm_output.squeeze(1)
        mse_loss = self.mse_loss(prediction, label)
        rating_loss = self.mse_loss_2(prediction, label)

        #import pdb; pdb.set_trace()
        return prediction, rating_loss 