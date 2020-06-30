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
        #import pdb; pdb.set_trace()

        return hidden_state

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)

        self.dropout = nn.Dropout(conf.dropout)

        self.reinit()

    def reinit(self):
        torch.manual_seed(0); 
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, input_vector, hidden_state=None):
        input_vector = self.dropout(input_vector)

        if hidden_state == None:
            output, hidden_state = self.rnn(input_vector)
        else:
            output, hidden_state = self.rnn(input_vector, hidden_state)
        return output, hidden_state

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
        hidden_state = self.encoder(user, item)

        outputs = []
        for t_input in review_input:
            input_vector = self.word_embedding(t_input.view(1, -1))
            output, hidden_state = self.decoder(input_vector, hidden_state)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).view(-1, conf.hidden_dim) # (tiem*batch, hidden_size)

        word_probit = self.rnn_out_linear(outputs) # (time*batch, vocab_sz)
        
        #import pdb; pdb.set_trace()
        out_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        obj = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD)

        return out_loss, obj
    
    def _sample_text_by_top_one(self, user, item, review_input):
        hidden_state = self.encoder(user, item)

        #import pdb; pdb.set_trace()

        next_word_idx = review_input[0]

        #import pdb; pdb.set_trace()
        sample_idx_list = [next_word_idx]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx).reshape(1, user.shape[0], -1)

            output, hidden_state = self.decoder(input_vector, hidden_state)
            word_probit = self.rnn_out_linear(output).reshape(-1, conf.vocab_sz)

            next_word_idx = torch.argmax(word_probit, 1)
                
            sample_idx_list.append(next_word_idx)
        
        sample_idx_list = torch.stack(sample_idx_list, dim=0).transpose(0, 1)
        return sample_idx_list
