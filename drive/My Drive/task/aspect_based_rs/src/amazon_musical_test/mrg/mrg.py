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

import config_mrg as conf 


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        torch.manual_seed(0); self.user_embedding = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.item_embedding = nn.Embedding(conf.num_items, conf.mf_dim)

        torch.manual_seed(0); 
        self.linears = []
        for idx in range(1, len(conf.mlp_dim_list)):
            self.linears.append(nn.Linear(conf.mlp_dim_list[idx-1], conf.mlp_dim_list[idx], bias=False).cuda())

        torch.manual_seed(0); 
        self.hidden_layer = nn.Linear(conf.mlp_embed_dim, conf.hidden_dim)

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)

        mlp_concat_emebd = torch.cat([user_embed, item_embed], dim=1)
        for idx in range(len(conf.mlp_dim_list)-1):
            mlp_concat_emebd = torch.tanh(self.linears[idx](mlp_concat_emebd))

        hidden_state = torch.tanh(self.hidden_layer(mlp_concat_emebd))\
            .view(1, -1, conf.hidden_dim) # (1, batch_size, hidden_dimension)


        '''1_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### FIRST PART #### '''
        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda() ### '''
        #### ****** verify review generation with GRU ****** ------ END ####

        return hidden_state, mlp_concat_emebd,

class decoder_rating(nn.Module):
    def __init__(self):
        super(decoder_rating, self).__init__()
        self.final_linear = nn.Linear(conf.mlp_dim_list[-1], 1)
        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)

        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.uniform_(self.final_linear.weight, -0.05, 0.05)
        torch.manual_seed(0); nn.init.constant_(self.final_linear.bias, 0.0)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, mlp_concat_emebd, user, item):
        pred = self.final_linear(mlp_concat_emebd) + conf.avg_rating + self.user_bias(user) + self.item_bias(item)
        return pred.view(-1)

class decoder_review(nn.Module):
    def __init__(self):
        super(decoder_review, self).__init__()
        self.rnn = nn.GRU(conf.word_dim+1*conf.mlp_dim_list[-1], conf.hidden_dim, num_layers=1)
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


class mrg(nn.Module): 
    def __init__(self):
        super(mrg, self).__init__()
        
        self.encoder = encoder()
        self.decoder_rating = decoder_rating()
        self.decoder_review = decoder_review()

        torch.manual_seed(0); self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        
        
        '''1_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### FIRST PART #### 
        '''
        torch.manual_seed(0); self.embedding_user = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.embedding_item = nn.Embedding(conf.num_items, conf.mf_dim)
        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)
        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda() ### 
        '''
        #### ****** veriify rating prediction with PMF ****** ------ END ####


        self.reinit()
        
    def reinit(self):
        '''2_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### SECOND PART #### 
        '''
        self.embedding_user.weight = torch.nn.Parameter(0.1 * self.embedding_user.weight)
        self.embedding_item.weight = torch.nn.Parameter(0.1 * self.embedding_item.weight)
        self.user_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_users, 1))
        self.item_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_items, 1)) ### 
        '''
        #### ****** veriify rating prediction with PMF ****** ------ END ####
        pass
        
    def forward(self, user, item, label, review_input, review_target):
        hidden_state, mlp_concat_emebd, = self.encoder(user, item)

        x_word_probit = []
        for t_input in review_input:
            input_vector = self.word_embedding(t_input.view(1, -1))
            input_vector = torch.cat([input_vector, mlp_concat_emebd.view(1, user.shape[0], -1)], dim=2)

            slice_word_probit, hidden_state = self.decoder_review(input_vector, hidden_state)
            x_word_probit.append(slice_word_probit)

        word_probit = torch.cat(x_word_probit, dim=0)

        pred = self.decoder_rating(mlp_concat_emebd, user, item)
        

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
        pred = x_prediction.view(-1)  ### 
        '''
        #### START ------ ****** veriify rating prediction with PMF ****** ####


        rating_out_loss = F.mse_loss(pred, label.view(-1), reduction='none')
        rating_obj_loss = F.mse_loss(pred, label.view(-1), reduction='sum')
        
        review_out_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        review_obj_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD)

        obj = 1e-7 * rating_obj_loss + 1.0 * review_obj_loss

        return rating_out_loss, review_out_loss, obj
    
    def predict_rating(self, user, item, label):
        _, mlp_concat_emebd, = self.encoder(user, item)

        pred = self.decoder_rating(mlp_concat_emebd, user, item)


        '''4_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### FOURTH PART #### 
        '''
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        output_emb = user_emb * item_emb
        x_prediction = torch.sum(output_emb, 1, keepdims=True) + self.avg_rating + user_bias + item_bias
        pred = x_prediction.view(-1)  ### 
        '''
        #### START ------ ****** veriify rating prediction with PMF ****** ####

        rating_out_loss = F.mse_loss(pred, label.view(-1), reduction='none')

        return pred, rating_out_loss
    
    def _sample_text_by_top_one(self, user, item, review_input):
        hidden_state, mlp_concat_emebd = self.encoder(user, item)
        mlp_concat_emebd = mlp_concat_emebd.view(1, -1, conf.mlp_dim_list[-1])

        next_word_idx = review_input[0]

        sample_idx_list = [next_word_idx]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx).reshape(1, -1, conf.word_dim)
            input_vector = torch.cat([input_vector, mlp_concat_emebd], dim=2)

            slice_word_probit, hidden_state = self.decoder_review(input_vector, hidden_state)
            word_probit = slice_word_probit
            next_word_idx = torch.argmax(word_probit, 1)
            sample_idx_list.append(next_word_idx)

        sample_idx_list = torch.stack(sample_idx_list, dim=0).transpose(0, 1)
        return sample_idx_list