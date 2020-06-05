import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import config_expansion_net as conf 

PAD = 0; SOS = 1; EOS = 2

class encoder(nn.Module):
    def __init__(self, word_embedding):
        super(encoder, self).__init__()

        self.word_embedding = word_embedding

        self.gamma_user_embedding = nn.Embedding(conf.num_users, conf.att_dim)
        self.gamma_item_embedding = nn.Embedding(conf.num_items, conf.att_dim)

        self.beta_user_embedding = nn.Embedding(conf.num_users, conf.aspect_dim)
        self.beta_item_embedding = nn.Embedding(conf.num_items, conf.aspect_dim)

        self.linear_eq_3 = nn.Linear(2*conf.att_dim, conf.hidden_dim)
        self.linear_eq_4 = nn.Linear(2*conf.aspect_dim, conf.hidden_dim)

        self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)

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

    def forward(self, user, item, summary):
        gamma_u = self.gamma_user_embedding(user) # (batch_size, att_dim)
        gamma_i = self.gamma_item_embedding(item) # (batch_size, att_dim)

        beta_u = self.beta_user_embedding(user) # (batch_size, aspect_dim)
        beta_i = self.beta_item_embedding(item) # (batch_size, aspect_dim)

        u_vector = torch.tanh(self.linear_eq_3(torch.cat([gamma_u, gamma_i], 1))) # (batch_size, hidden_dim)
        v_vector = torch.tanh(self.linear_eq_4(torch.cat([beta_u, beta_i], 1))) # (batch_size, hidden_dim)

        summary_emb = self.word_embedding(summary) #(seq_len, batch, word_dim)
        summary_emb = F.dropout(summary_emb, p=conf.dropout)
        outputs, _ = self.rnn(summary_emb) #(seq_len, batch, hidden_dim)
        
        #import pdb; pdb.set_trace()

        hidden_state = (u_vector + v_vector + outputs[-1]).view(\
            1, user.shape[0], conf.hidden_dim) # (1, batch, hidden_size=n)

        #hidden_state = (u_vector + v_vector).view(1, user.shape[0], conf.hidden_dim) # (1, batch, hidden_size=n)

        return outputs, gamma_u, gamma_i, beta_u, beta_i, hidden_state

class decoder(nn.Module):
    def __init__(self, word_embedding):
        super(decoder, self).__init__()
        self.word_embedding = word_embedding
        self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)

        self.linear_eq_8 = nn.Linear(2*conf.hidden_dim, 1)
        self.linear_eq_10 = nn.Linear(conf.att_dim+conf.hidden_dim, 1)
        self.linear_eq_11 = nn.Linear(2*conf.aspect_dim, conf.aspect_dim)
        self.linear_eq_12 = nn.Linear(conf.aspect_dim+conf.word_dim+conf.hidden_dim, conf.aspect_dim)
        #self.linear_eq_13 = nn.Linear(2*conf.hidden_dim+conf.att_dim, conf.vocab_sz)
        self.linear_eq_13 = nn.Linear(conf.hidden_dim, conf.vocab_sz)

        self.linear_x = nn.Linear(conf.hidden_dim, conf.vocab_sz)

        self.reinit()

    def reinit(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.manual_seed(0); nn.init.xavier_normal_(param.data)
            else:
                nn.init.zeros_(param.data)
        
        nn.init.zeros_(self.linear_eq_8.bias)
        torch.manual_seed(0); nn.init.xavier_normal_(self.linear_eq_8.weight)
        nn.init.zeros_(self.linear_eq_10.bias)
        torch.manual_seed(0); nn.init.xavier_normal_(self.linear_eq_10.weight)
        nn.init.zeros_(self.linear_eq_11.bias)
        torch.manual_seed(0); nn.init.xavier_normal_(self.linear_eq_11.weight)
        nn.init.zeros_(self.linear_eq_12.bias)
        torch.manual_seed(0); nn.init.xavier_normal_(self.linear_eq_12.weight)
        nn.init.zeros_(self.linear_eq_13.bias)
        torch.manual_seed(0); nn.init.xavier_normal_(self.linear_eq_13.weight)

        nn.init.zeros_(self.linear_x.bias)
        torch.manual_seed(0); nn.init.xavier_normal_(self.linear_x.weight)

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
            encoder_outputs, 
            summary,
            gamma_u,
            gamma_i,
            beta_u, 
            beta_i, 
            review_aspect, 
            review_aspect_mask,
        ):
        input_vector = F.dropout(input_vector, p=conf.dropout)

        _, hidden_state = self.rnn(input_vector, hidden_state)

        # calculate a^1_t
        mask = (summary > 0).long()
        a_1_t = self._attention(encoder_outputs, hidden_state, mask, self.linear_eq_8) # (batch, hidden_dim)

        # calculate a^2_t
        mask = torch.ones(2, gamma_u.shape[0]).cuda()
        a_2_t = self._attention(torch.stack([gamma_u, gamma_i], dim=0), hidden_state, mask, self.linear_eq_10) #(batch, att_dim)

        # calculate a^3_t
        s_ui = self.linear_eq_11(torch.cat([beta_u, beta_i], dim=1)) # (batch, aspect_dim)
        #a_3_t = torch.tanh(self.linear_eq_12(torch.cat([s_ui, \
        #    input_vector.view(-1, conf.word_dim), hidden_state.view(-1, conf.hidden_dim)], dim=1))) # (batch, aspect_dim)
        
        a_3_t = (self.linear_eq_12(torch.cat([s_ui, \
            input_vector.view(-1, conf.word_dim), hidden_state.view(-1, conf.hidden_dim)], dim=1))) # (batch, aspect_dim)

        #PvWt = torch.tanh(self.linear_eq_13(torch.cat([hidden_state.view(-1, conf.hidden_dim), a_1_t, a_2_t], dim=1))) # (batch, vocab_size)

        aspect_probit = torch.index_select(a_3_t, 1, review_aspect) * review_aspect_mask # (seq_length*batch_size, vocab_sz)

        word_probit = self.linear_x(hidden_state.view(-1, conf.hidden_dim)) # (batch, vocab_sz)

        #return PvWt + aspect_probit, hidden_state
        return word_probit + aspect_probit, hidden_state

class expansion_net(nn.Module):
    def __init__(self):
        super(expansion_net, self).__init__()

        # PARAMETERS FOR LSTM
        torch.manual_seed(0); self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        
        self.encoder = encoder(self.word_embedding)
        self.decoder = decoder(self.word_embedding)

    def forward(self, user, item, summary, review_input, review_output, review_aspect, review_aspect_mask):
        encoder_outputs, gamma_u, gamma_i, beta_u, beta_i, hidden_state = \
            self.encoder(user, item, summary)

        #hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda() # (1, 1, hidden_dim)

        Pwt = []
        for t in range(conf.rev_len):
            input_vector = self.word_embedding(review_input[t]).view(1, -1, conf.word_dim) # (1, batch, hidden_dim)
            total_word_probit, hidden_state = self.decoder(
                input_vector, 
                hidden_state,
                encoder_outputs, 
                summary,
                gamma_u,
                gamma_i,
                beta_u, 
                beta_i, 
                review_aspect, 
                review_aspect_mask,
            )

            Pwt.append(total_word_probit)

        Pwt = torch.cat(Pwt, dim=0)
        obj_loss = F.cross_entropy(Pwt, review_output.reshape(-1))

        return obj_loss

    def _sample_text_by_top_one(self, user, item, summary, review_input, review_aspect, review_aspect_mask):
        encoder_outputs, gamma_u, gamma_i, beta_u, beta_i, hidden_state = \
            self.encoder(user, item, summary)
        

        #hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda() # (1, 1, hidden_dim)

        next_word_idx = review_input[0][0].view(1, 1)
        sample_idx_list = [next_word_idx.item()]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx).reshape(1, 1, -1)

            total_word_probit, hidden_state = self.decoder(
                input_vector, 
                hidden_state,
                encoder_outputs, 
                summary,
                gamma_u,
                gamma_i,
                beta_u, 
                beta_i, 
                review_aspect, 
                review_aspect_mask,
            )

            next_word_idx = torch.argmax(total_word_probit, 1)
            if next_word_idx.item() == PAD:
                return sample_idx_list
                
            sample_idx_list.append(next_word_idx.item())
        return sample_idx_list