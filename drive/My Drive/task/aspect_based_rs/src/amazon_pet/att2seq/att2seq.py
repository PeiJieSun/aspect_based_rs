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

import config_att2seq as conf 

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        torch.manual_seed(0); self.user_embedding = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.item_embedding = nn.Embedding(conf.num_items, conf.mf_dim)

        self.hidden_layer = nn.Linear(conf.mf_dim, conf.hidden_dim)
    
        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)

        hidden_state = torch.tanh(self.hidden_layer(user_embed+item_embed))\
            .view(1, -1, conf.hidden_dim) # (1, batch_size, hidden_dimension)


        '''1_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### FIRST PART #### 
        '''
        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda() ### 
        '''
        #### ****** verify review generation with GRU ****** ------ END ####

        return hidden_state, user_embed, item_embed

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)

        self.dropout = nn.Dropout(conf.dropout)
        self.linear_eq_10 = nn.Linear(conf.mf_dim+conf.hidden_dim, 1)
        self.linear_eq_13 = nn.Linear(conf.hidden_dim+1*conf.mf_dim, conf.vocab_sz)

        ''''ATTENTION PLEASE!!!'''
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

        nn.init.zeros_(self.linear_eq_10.bias)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.linear_eq_10.weight)

        nn.init.zeros_(self.linear_eq_13.bias)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.linear_eq_13.weight)

        ''''ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### THIRD PART #### '''
        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias) ### '''
        #### ****** verify review generation with GRU ****** ------ END ####

    # value: (seq_len, batch, hidden_dim)
    # query: (1, batch, hidden_dim)
    def _attention(self, value, query, mask, func):
        query = query.repeat(value.shape[0], 1, 1) # (seq_len, batch, hidden_dim) 
        key = torch.cat([value, query], dim=2)  # (seq_len, batch, hidden_dim*2)
        
        a_1 = torch.tanh(func(key)).view(value.shape[0], -1)  # (seq_len, batch)
        a_2 = torch.exp(a_1) #  (seq_len, batch)
        a_3 = a_2 * mask # (seq_len, batch)
        a_4 = torch.sum(a_3, dim=0, keepdim=True) + 1e-6 # (1, batch)
        a_5 = (a_3 / a_4).view(value.shape[0], -1, 1) # (seq_len, batch, 1)
        a_t = torch.sum(a_5 * value, dim=0) # (batch, hidden_dim)

        return a_t

    def forward(self, input_vector, hidden_state, user_embed, item_embed):
        input_vector = self.dropout(input_vector)

        output, hidden_state = self.rnn(input_vector, hidden_state)

        mask = torch.ones(2, user_embed.shape[0]).cuda()
        a_2_t = self._attention(torch.stack([user_embed, item_embed], dim=0), \
            hidden_state, mask, self.linear_eq_10) #(batch, att_dim)
        
        word_probit = self.linear_eq_13(torch.cat(\
            [hidden_state.view(-1, conf.hidden_dim), a_2_t], dim=1)) # (batch, vocab_size)

        ''''ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### FOURTH PART #### 
        '''
        word_probit = self.rnn_out_linear(hidden_state.view(-1, conf.hidden_dim)) ### 
        '''
        #### ****** verify review generation with GRU ****** ------ END ####

        return word_probit, hidden_state

class att2seq(nn.Module): 
    def __init__(self):
        super(att2seq, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()

        torch.manual_seed(0); self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        self.rnn_out_linear = nn.Linear(conf.hidden_dim, conf.vocab_sz)
        
        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias)
    
    def forward(self, user, item, review_input, review_target):
        hidden_state, user_embed, item_embed = self.encoder(user, item)


        x_word_probit = []
        for t_input in review_input:
            input_vector = self.word_embedding(t_input.view(1, -1))
            slice_word_probit, hidden_state = self.decoder(input_vector, hidden_state, user_embed, item_embed)
            x_word_probit.append(slice_word_probit)
        word_probit = torch.cat(x_word_probit, dim=0)
        

        out_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        obj = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD)

        return out_loss, obj
    
    def _sample_text_by_top_one(self, user, item, review_input):
        hidden_state, user_embed, item_embed = self.encoder(user, item)


        next_word_idx = review_input[0]
        sample_idx_list = [next_word_idx]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx).reshape(1, user.shape[0], -1)

            slice_word_probit, hidden_state = self.decoder(input_vector, hidden_state, user_embed, item_embed)
            word_probit = slice_word_probit
            next_word_idx = torch.argmax(word_probit, 1)
            sample_idx_list.append(next_word_idx)


        sample_idx_list = torch.stack(sample_idx_list, dim=0).transpose(0, 1)
        return sample_idx_list