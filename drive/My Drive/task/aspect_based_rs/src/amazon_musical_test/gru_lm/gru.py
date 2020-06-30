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

import config_gru as conf 

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        torch.manual_seed(0); self.hidden_layer = nn.Linear(conf.mf_dim, conf.hidden_dim)
    
        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)

    def forward(self, user, item):

        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda()
        return hidden_state

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        #self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)
        self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1)

        self.dropout = nn.Dropout(conf.dropout)

        torch.manual_seed(0); self.rnn_out_linear = nn.Linear(conf.hidden_dim, conf.vocab_sz)
        
        self.reinit()

    def reinit(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.manual_seed(0); nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias)

    def forward(self, input_vector, hidden_state=None):
        input_vector = self.dropout(input_vector)

        if hidden_state == None:
            output, hidden_state = self.rnn(input_vector)
        else:
            output, hidden_state = self.rnn(input_vector, hidden_state)

        word_probit = self.rnn_out_linear(hidden_state.view(-1, conf.hidden_dim)) # (batch, vocab_sz)

        return word_probit, hidden_state

class gru(nn.Module): 
    def __init__(self):
        super(gru, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()

        torch.manual_seed(0); self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        torch.manual_seed(0); self.rnn_out_linear = nn.Linear(conf.hidden_dim, conf.vocab_sz)
        
        torch.manual_seed(0); self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1)

        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias)
    
    # user, item, review_input, review_target
    def forward(self, *values):
        user, item, review_input, review_target = values[0], values[1], values[2], values[3]
        hidden_state = self.encoder(user, item)

        x_word_probit = []

        for t_input in review_input:
            input_vector = self.word_embedding(t_input.view(1, -1))

            slice_word_probit, hidden_state = self.decoder(input_vector, hidden_state)

            x_word_probit.append(slice_word_probit)
        
        word_probit = torch.cat(x_word_probit, dim=0)

        out_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        obj = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD)

        return out_loss, obj

    def _sample_text_by_top_one(self, *args):
        user, item, review_input = args[0], args[1], args[2]

        hidden_state = self.encoder(user, item)

        #import pdb; pdb.set_trace()

        next_word_idx = review_input[0]

        #import pdb; pdb.set_trace()
        sample_idx_list = [next_word_idx]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx.view(1, -1))

            slice_word_probit, hidden_state = self.decoder(input_vector, hidden_state)
            word_probit = slice_word_probit
            
            next_word_idx = torch.argmax(word_probit, 1)
            sample_idx_list.append(next_word_idx)

        sample_idx_list = torch.stack(sample_idx_list, 1)#.reshape(-1, conf.rev_len)
        
        return sample_idx_list