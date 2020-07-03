import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import config_expansion_net as conf 

PAD = 0; SOS = 1; EOS = 2

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.gamma_user_embedding = nn.Embedding(conf.num_users, conf.att_dim)
        self.gamma_item_embedding = nn.Embedding(conf.num_items, conf.att_dim)

        self.beta_user_embedding = nn.Embedding(conf.num_users, conf.aspect_dim)
        self.beta_item_embedding = nn.Embedding(conf.num_items, conf.aspect_dim)

        self.linear_eq_3 = nn.Linear(2*conf.att_dim, conf.hidden_dim)
        self.linear_eq_4 = nn.Linear(2*conf.aspect_dim, conf.hidden_dim)

        #self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)
        self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1)

        self.dropout = nn.Dropout(conf.dropout)
        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_normal_(self.gamma_user_embedding.weight)
        torch.manual_seed(0); nn.init.xavier_normal_(self.gamma_item_embedding.weight)
        torch.manual_seed(0); nn.init.xavier_normal_(self.beta_user_embedding.weight)
        torch.manual_seed(0); nn.init.xavier_normal_(self.beta_item_embedding.weight)

        torch.manual_seed(0); nn.init.xavier_normal_(self.linear_eq_3.weight)
        torch.manual_seed(0); nn.init.xavier_normal_(self.linear_eq_4.weight)
        nn.init.zeros_(self.linear_eq_3.bias)
        nn.init.zeros_(self.linear_eq_4.bias)

        # initialize the parameters in self.rnn
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.manual_seed(0); nn.init.xavier_normal_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, user, item):
        gamma_u = self.gamma_user_embedding(user) # (batch_size, att_dim)
        gamma_i = self.gamma_item_embedding(item) # (batch_size, att_dim)

        beta_u = self.beta_user_embedding(user) # (batch_size, aspect_dim)
        beta_i = self.beta_item_embedding(item) # (batch_size, aspect_dim)

        u_vector = torch.tanh(self.linear_eq_3(torch.cat([gamma_u, gamma_i], 1))) # (batch_size, hidden_dim)
        v_vector = torch.tanh(self.linear_eq_4(torch.cat([beta_u, beta_i], 1))) # (batch_size, hidden_dim)
        
        hidden_state = (u_vector + v_vector).view(1, user.shape[0], conf.hidden_dim) # (1, batch, hidden_size=n)
        #hidden_state = (u_vector).view(1, user.shape[0], conf.hidden_dim) # (1, batch, hidden_size=n)


        '''1_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### FIRST PART #### 
        '''
        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda() ### 
        '''
        #### ****** verify review generation with GRU ****** ------ END ####

        return gamma_u, gamma_i, beta_u, beta_i, hidden_state


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        #self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)
        self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1)

        
        self.linear_eq_10 = nn.Linear(conf.att_dim+conf.hidden_dim, 1)
        
        self.linear_eq_11 = nn.Linear(2*conf.aspect_dim, conf.aspect_dim)
        
        
        self.linear_eq_12 = nn.Linear(1*conf.aspect_dim+1*conf.word_dim+1*conf.hidden_dim, conf.aspect_dim)
        self.linear_eq_13 = nn.Linear(conf.hidden_dim+1*conf.att_dim, conf.vocab_sz)

        self.linear_x = nn.Linear(conf.hidden_dim, conf.vocab_sz)

        self.dropout = nn.Dropout(conf.dropout)


        '''2_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### SECOND PART #### 
        '''
        torch.manual_seed(0); self.rnn_out_linear = nn.Linear(conf.hidden_dim, conf.vocab_sz) ### 
        '''
        #### ****** verify review generation with GRU ****** ------ END ####


        self.reinit()

    def reinit(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.manual_seed(0); nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)
        
        
        nn.init.zeros_(self.linear_eq_11.bias)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.linear_eq_11.weight)
        
        

        nn.init.zeros_(self.linear_eq_12.bias)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.linear_eq_12.weight)

        nn.init.zeros_(self.linear_eq_10.bias)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.linear_eq_10.weight)

        nn.init.zeros_(self.linear_eq_13.bias)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.linear_eq_13.weight)

        nn.init.zeros_(self.linear_x.bias)
        torch.manual_seed(0); nn.init.xavier_uniform_(self.linear_x.weight)



        '''3_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### THIRD PART #### 
        '''
        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias) ### 
        '''
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

    def forward(self, 
            input_vector, 
            hidden_state, 
            gamma_u,
            gamma_i,
            beta_u, 
            beta_i, 
            review_aspect_mask,
        ):
        input_vector = self.dropout(input_vector)

        output, hidden_state = self.rnn(input_vector, hidden_state)

        
        # calculate a^2_t
        mask = torch.ones(2, gamma_u.shape[0]).cuda()
        a_2_t = self._attention(torch.stack([gamma_u, gamma_i], dim=0), hidden_state, mask, self.linear_eq_10) #(batch, att_dim)

        
        # calculate a^3_t
        s_ui = self.linear_eq_11(torch.cat([beta_u, beta_i], dim=1)) # (batch, aspect_dim)
        #a_3_t = torch.tanh(self.linear_eq_12(torch.cat([s_ui, \
        #    input_vector.view(-1, conf.word_dim), hidden_state.view(-1, conf.hidden_dim)], dim=1))) # (batch, aspect_dim)
        
        a_3_t = (self.linear_eq_12(torch.cat([s_ui, hidden_state.view(-1, conf.hidden_dim),\
            input_vector.view(-1, conf.word_dim)], dim=1))) # (batch, aspect_dim)

        #a_3_t = (self.linear_eq_12(torch.cat([s_ui], dim=1))) # (batch, aspect_dim)
        
        #PvWt = torch.tanh(self.linear_eq_13(torch.cat([hidden_state.view(-1, conf.hidden_dim), a_1_t, a_2_t], dim=1))) # (batch, vocab_size)
        PvWt = self.linear_eq_13(torch.cat([hidden_state.view(-1, conf.hidden_dim), a_2_t], dim=1)) # (batch, vocab_size)

        #PvWt = self.linear_eq_13(torch.cat([hidden_state.view(-1, conf.hidden_dim)], dim=1)) # (batch, vocab_size)

        #import pdb; pdb.set_trace()
        #aspect_probit = torch.index_select(a_3_t, 1, review_aspect) * review_aspect_mask # (seq_length*batch_size, vocab_sz)
        
        x_aspect_probit = torch.sparse.mm(review_aspect_mask, a_3_t.t()).t() #batch, vocab_sz
        #import  pdb; pdb.set_trace()

        word_probit = self.linear_x(hidden_state.view(-1, conf.hidden_dim)) # (batch, vocab_sz)


        '''4_REVIEW GENERATION ATTENTION PLEASE!!!'''
        #### START ------ ****** verify review generation with GRU ****** ####
        #### FOURTH PART #### 
        '''
        word_probit = self.rnn_out_linear(hidden_state.view(-1, conf.hidden_dim)) ### 
        '''
        #### ****** verify review generation with GRU ****** ------ END ####


        return PvWt + 1.0*x_aspect_probit, hidden_state
        #return word_probit, hidden_state
        #return PvWt, hidden_state

class expansion_net(nn.Module): 
    def __init__(self):
        super(expansion_net, self).__init__()

        torch.manual_seed(0); self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        torch.manual_seed(0); self.rnn_out_linear = nn.Linear(conf.hidden_dim, conf.vocab_sz)
        
        self.encoder = encoder()
        self.decoder = decoder()

        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias)
    
    # user, item, review_input, review_target
    def forward(self, user, item, review_input, review_target, review_aspect_mask):
        #user, item, review_input, review_target = values[0], values[1], values[3], values[4]
        gamma_u, gamma_i, beta_u, beta_i, hidden_state = self.encoder(user, item)

        outputs = []

        x_word_probit = []

        for t_input in review_input:
            input_vector = self.word_embedding(t_input.view(1, -1))
            #output, hidden_state = self.decoder(input_vector, hidden_state)

            slice_word_probit, hidden_state = self.decoder(input_vector, 
                hidden_state, 
                gamma_u,
                gamma_i,
                beta_u, 
                beta_i, 
                review_aspect_mask
            )

            x_word_probit.append(slice_word_probit)
        
        word_probit = torch.cat(x_word_probit, dim=0)

        #import pdb; pdb.set_trace()
        out_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        obj = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD)

        return out_loss, obj
    
    def _sample_text_by_top_one(self, user, item, review_input, review_aspect_mask):
        gamma_u, gamma_i, beta_u, beta_i, hidden_state = self.encoder(user, item)

        x_user, x_item = set(tensorToScalar(user).tolist()), set(tensorToScalar(item).tolist())

        #import pdb; pdb.set_trace()

        next_word_idx = review_input[0]

        #import pdb; pdb.set_trace()
        sample_idx_list = [next_word_idx]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx.view(1, -1))

            slice_word_probit, hidden_state = self.decoder(
                input_vector, 
                hidden_state, 
                gamma_u,
                gamma_i,
                beta_u, 
                beta_i, 
                review_aspect_mask
            )

            word_probit = slice_word_probit

            next_word_idx = torch.argmax(word_probit, 1)
            sample_idx_list.append(next_word_idx)
        
        #import pdb; pdb.set_trace()
        sample_idx_list = torch.stack(sample_idx_list, 1)#.reshape(-1, conf.rev_len)
        
        return sample_idx_list