'''
    Description: This code is named language model, which can genreate texts based on word-level
    The input of this model is the real reviews, and each output at each time is just be influenced by the previous words

    The review generation process considers the user-item interaction information and rating information
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

import config_cf_gcn as conf 

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        torch.manual_seed(0); self.gamma_user_embedding = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.gamma_item_embedding = nn.Embedding(conf.num_items, conf.mf_dim)

        torch.manual_seed(0); self.theta_user_embedding = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.theta_item_embedding = nn.Embedding(conf.num_items, conf.mf_dim)

        torch.manual_seed(0); self.doc_embedding = nn.Embedding(conf.num_words, conf.encoder_word_dim)

        self.reinit()

    def reinit(self):
        self.gamma_user_embedding.weight = torch.nn.Parameter(0.1*self.gamma_user_embedding.weight)
        self.gamma_item_embedding.weight = torch.nn.Parameter(0.1*self.gamma_item_embedding.weight)

        self.theta_user_embedding.weight = torch.nn.Parameter(0.1*self.theta_user_embedding.weight)
        self.theta_item_embedding.weight = torch.nn.Parameter(0.1*self.theta_item_embedding.weight)

    def forward(self, user, item, user_doc, item_doc):
        gamma_user_embed = self.gamma_user_embedding(user)
        gamma_item_embed = self.gamma_item_embedding(item)

        theta_user_embed = self.theta_user_embedding(user)
        theta_item_embed = self.theta_item_embedding(item)

        user_doc_embed = \
            self.doc_embedding(user_doc).view(-1, conf.seq_len*conf.user_seq_num, conf.encoder_word_dim)
        item_doc_embed = \
            self.doc_embedding(item_doc).view(-1, conf.seq_len*conf.item_seq_num, conf.encoder_word_dim)

        user_doc_embed = torch.mean(user_doc_embed, dim=1)
        item_doc_embed = torch.mean(item_doc_embed, dim=1)

        '''1_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### FIRST PART #### '''
        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda() ### '''
        #### ****** verify review generation with GRU ****** ------ END ####

        return [
            gamma_user_embed, 
            gamma_item_embed, 
            theta_user_embed, 
            theta_item_embed, 
            user_doc_embed, 
            item_doc_embed,
            hidden_state
        ]

class decoder_rating(nn.Module):
    def __init__(self):
        super(decoder_rating, self).__init__()

        self.transform_user = nn.Linear(conf.encoder_word_dim, conf.mf_dim, bias=False)
        self.transform_item = nn.Linear(conf.encoder_word_dim, conf.mf_dim, bias=False)

        self.predict_layer = nn.Linear(3*conf.mf_dim, 1)

        self.b_users = nn.Parameter(torch.randn(conf.num_users, 1))
        self.b_items = nn.Parameter(torch.randn(conf.num_items, 1))

        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.uniform_(self.transform_user.weight, -0.1, 0.1)
        torch.manual_seed(0); nn.init.uniform_(self.transform_item.weight, -0.1, 0.1)

        torch.manual_seed(0); nn.init.uniform_(self.predict_layer.weight, -0.05, 0.05)
        nn.init.constant_(self.predict_layer.bias, 0.0)

        torch.manual_seed(0); nn.init.uniform_(self.b_users, a=0, b=0.1)
        torch.manual_seed(0); nn.init.uniform_(self.b_items, a=0, b=0.1)

    def forward(self, user, item, embed):
        [gamma_user_embed, 
            gamma_item_embed, 
            theta_user_embed, 
            theta_item_embed, 
            user_doc_embed, 
            item_doc_embed] = embed

        concat_embed_1 = gamma_user_embed * gamma_item_embed    
        concat_embed_2 = theta_user_embed * self.transform_user(user_doc_embed)
        concat_embed_3 = theta_item_embed * self.transform_item(item_doc_embed)

        mlp_concat_embed = torch.cat([concat_embed_1, concat_embed_2, concat_embed_3], dim=1)

        #import pdb; pdb.set_trace()

        pred = self.predict_layer(mlp_concat_embed) + conf.avg_rating \
            + self.b_users[user] + self.b_items[item]

        return pred.view(-1)

class decoder_review(nn.Module):
    def __init__(self):
        super(decoder_review, self).__init__()
        self.rnn = nn.GRU(conf.word_dim+0*conf.mf_dim, conf.hidden_dim, num_layers=1)
        
        self.dropout = nn.Dropout(conf.dropout)


        '''2_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### SECOND PART #### '''
        torch.manual_seed(0); self.rnn_out_linear = nn.Linear(conf.hidden_dim, conf.vocab_sz) ### '''
        #### ****** verify review generation with GRU ****** ------ END ####
        
        self.reinit()

    def reinit(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.manual_seed(0); nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

        '''3_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### THIRD PART #### '''
        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias) ### '''
        #### ****** verify review generation with GRU ****** ------ END ####

    def forward(self, input_vector, hidden_state):
        input_vector = self.dropout(input_vector)

        output, hidden_state = self.rnn(input_vector, hidden_state)


        '''4_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### FOURTH PART #### '''
        word_probit = self.rnn_out_linear(hidden_state.view(-1, conf.hidden_dim)) ### '''
        #### ****** verify review generation with GRU ****** ------ END ####

        return word_probit, hidden_state

class cf_gcn(nn.Module):
    def __init__(self):
        super(cf_gcn, self).__init__()

        self.encoder = encoder()
        self.decoder_rating = decoder_rating()
        self.decoder_review = decoder_review()

        torch.manual_seed(0); self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        
        '''1_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### FIRST PART #### '''
        torch.manual_seed(0); self.embedding_user = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.embedding_item = nn.Embedding(conf.num_items, conf.mf_dim)
        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)
        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda() ### '''
        #### ****** veriify rating prediction with PMF ****** ------ END ####
        
        self.reinit()

    def reinit(self):

        '''2_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### SECOND PART #### '''
        self.embedding_user.weight = torch.nn.Parameter(0.1 * self.embedding_user.weight)
        self.embedding_item.weight = torch.nn.Parameter(0.1 * self.embedding_item.weight)
        self.user_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_users, 1))
        self.item_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_items, 1)) ### '''
        #### ****** veriify rating prediction with PMF ****** ------ END ####
        
    
    # values: user, item, label, review_input, review_target, user_doc, item_doc
    def forward(self, user, item, label, review_input, review_target, user_doc, item_doc):

        embed = self.encoder(user, item, user_doc, item_doc)
        hidden_state = embed[-1]
        embed = embed[:-1]

        x_word_probit = []
        for t_input in review_input:
            input_vector = self.word_embedding(t_input).view(1, user.shape[0], -1)
            #input_vector = torch.cat([input_vector, embed[0], embed[1]], dim=1).view(1, user.shape[0], -1)

            slice_word_probit, hidden_state = self.decoder_review(input_vector, hidden_state)
            x_word_probit.append(slice_word_probit)

        word_probit = torch.cat(x_word_probit, dim=0)
        
        review_out_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        review_obj = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD)

        ###### rating prediction
        pred = self.decoder_rating(user, item, embed)

        '''3_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### THIRD PART #### '''
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        output_emb = user_emb * item_emb
        x_prediction = torch.sum(output_emb, 1, keepdims=True) + self.avg_rating + user_bias + item_bias
        pred = x_prediction.view(-1)  ### '''
        #### START ------ ****** veriify rating prediction with PMF ****** ####


        rating_out_loss = F.mse_loss(pred, label, reduction='none')
        rating_obj = F.mse_loss(pred, label, reduction='sum')

        obj = 1.0*rating_obj + 0.0*review_obj

        return review_out_loss, rating_out_loss, obj

    def predict_rating(self, values):
        [user, item, label, review_input, review_target, user_doc, item_doc] = values

        embed = self.encoder(user, item, user_doc, item_doc)
        embed = embed[:-1]

        ###### rating prediction
        pred = self.decoder_rating(user, item, embed)

        '''4_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### FOURTH PART #### '''
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        output_emb = user_emb * item_emb
        x_prediction = torch.sum(output_emb, 1, keepdims=True) + self.avg_rating + user_bias + item_bias
        pred = x_prediction.view(-1)  ### '''
        #### START ------ ****** veriify rating prediction with PMF ****** ####


        rating_out_loss = F.mse_loss(pred, label, reduction='none')

        return rating_out_loss

    def _sample_text_by_top_one(self, user, item, review_input, user_doc, item_doc):
        embed = self.encoder(user, item, user_doc, item_doc)
        hidden_state = embed[-1]
        embed = embed[:-1]

        next_word_idx = review_input[0]

        sample_idx_list = [next_word_idx]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx).view(1, user.shape[0], -1)
            #input_vector = torch.cat([input_vector, embed[0], embed[1]], dim=1).view(1, user.shape[0], -1)

            slice_word_probit, hidden_state = self.decoder_review(input_vector, hidden_state)
            word_probit = slice_word_probit
            next_word_idx = torch.argmax(word_probit, 1)
            sample_idx_list.append(next_word_idx)

        sample_idx_list = torch.stack(sample_idx_list, dim=0).transpose(0, 1)
        return sample_idx_list