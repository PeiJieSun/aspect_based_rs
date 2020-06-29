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

        torch.manual_seed(0); self.user_embedding = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.item_embedding = nn.Embedding(conf.num_items, conf.mf_dim)

        torch.manual_seed(0); self.hidden_layer = nn.Linear(conf.mf_dim, conf.hidden_dim)
    
        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)

    def forward(self, user, item):
        '''
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)

        hidden_state = torch.tanh(self.hidden_layer(user_embed+item_embed))\
            .view(1, -1, conf.hidden_dim) # (1, batch_size, hidden_dimension)
        '''
        #import pdb; pdb.set_trace()

        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda()
        return hidden_state

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        #self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)
        self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1)

        self.dropout = nn.Dropout(conf.dropout)

        self.reinit()

    def reinit(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.manual_seed(0); nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, input_vector, hidden_state=None):
        input_vector = self.dropout(input_vector)

        if hidden_state == None:
            output, hidden_state = self.rnn(input_vector)
        else:
            output, hidden_state = self.rnn(input_vector, hidden_state)
        return output, hidden_state

class gru(nn.Module): 
    def __init__(self):
        super(gru, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()

        torch.manual_seed(0); self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        torch.manual_seed(0); self.rnn_out_linear = nn.Linear(conf.hidden_dim, conf.vocab_sz)
        
        #self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1, dropout=conf.dropout)

        torch.manual_seed(0); self.rnn = nn.GRU(conf.word_dim, conf.hidden_dim, num_layers=1)

        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias)
    
    # user, item, review_input, review_target
    def forward(self, user, item, review_input, review_target):
        #user, item, review_input, review_target = values[0], values[1], values[3], values[4]
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
            input_vector = self.word_embedding(next_word_idx.view(1, -1))

            #import pdb; pdb.set_trace()
            output, hidden_state = self.decoder(input_vector, hidden_state)
            word_probit = self.rnn_out_linear(output).reshape(-1, conf.vocab_sz)

            
            next_word_idx = torch.argmax(word_probit, 1)
            sample_idx_list.append(next_word_idx)
        #import pdb; pdb.set_trace()
        sample_idx_list = torch.stack(sample_idx_list, 1)#.reshape(-1, conf.rev_len)
        
        return sample_idx_list
    '''
    
    def _sample_text_by_top_one(self, user, item, review_input):
        hidden_state = self.encoder(user, item)

        #import pdb; pdb.set_trace()

        next_word_idx = review_input[0][0].view(1, 1)

        #import pdb; pdb.set_trace()
        sample_idx_list = [next_word_idx.item()]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx).reshape(1, 1, -1)

            output, hidden_state = self.decoder(input_vector, hidden_state)
            word_probit = self.rnn_out_linear(output).reshape(-1, conf.vocab_sz)

            next_word_idx = torch.argmax(word_probit, 1)
            if next_word_idx.item() == PAD:
                return sample_idx_list
                
            sample_idx_list.append(next_word_idx.item())
        return sample_idx_list
    '''

    '''
    def _sample_text_by_beam_search(self, user, item, review_input):
        beam_size = 4

        previous_sequence = {}
        current_sequence = {}

        # initialize the dictionary which stores the generated words token idx
        for sub_idx in range(beam_size):
            previous_sequence[sub_idx] = [SOS]

        # first iteration: prepare the input text data
        input_vector = self.encoder(user, item)
        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda()
        output, hidden_state = self.decoder(input_vector, hidden_state)
        word_probit = self.rnn_out_linear(output).reshape(-1, conf.vocab_sz)

        # values: (batch_size, beam_size)
        values, indices = torch.topk(word_probit, beam_size) # indices: (batch_size, beam_size)

        # prepare the input data for next iteration
        next_word_idx = indices.view(1, -1) # (1, batch_size*beam_size)
        previous_probit = values.view(-1, 1) # (batch_size*beam_size, 1)

        for sub_idx in range(next_word_idx.shape[1]):
            previous_sequence[sub_idx].append(indices[0][sub_idx].item())

        #second iteration: construct the hidden_state for current iteration
        hidden_state = hidden_state.repeat(1, beam_size, 1)  # (1, beam_size, hidden_size)

        for _ in range(conf.rev_len):
            seed_vector = self.word_embedding(next_word_idx)

            # hidden_state: ((2, batch_size, hidden_size), (2, batch_size, hidden_size))
            outputs, hidden_state = self.decoder(seed_vector, hidden_state) # review_output_embedding: (batch_size*beam_size, word_dimension)
            word_probit = self.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)

            tmp_hidden_state = deepcopy(hidden_state.data)

            current_probit = word_probit + previous_probit # (batch_size*beam_size, vocab_size)
            
            first_values, first_indices = torch.topk(current_probit, beam_size) # (batch_size*beam_size, beam_size)

            first_values = first_values.view(-1) # (batch_size, beam_size*beam_size)
            first_indices = first_indices.view(-1) # (batch_size, beam_size*beam_size)

            # second_values: (batch_size, beam_size)
            second_values, second_indices = torch.topk(first_values, beam_size) # second_indices: (batch_size, beam_size)

            # collect the predicted words and store them to the dictionary
            for outer_idx, top_word_index in enumerate(second_indices):
                previous_probit[outer_idx] = second_values[outer_idx]
                next_word_idx[0][outer_idx] = first_indices[top_word_index]
                
                if first_indices[top_word_index] == PAD:
                    return current_sequence[0]
                current_sequence[outer_idx] = deepcopy(previous_sequence[int(top_word_index.item() / 4)])
                current_sequence[outer_idx].append(first_indices[top_word_index].item())
                hidden_state[0][outer_idx] = deepcopy(tmp_hidden_state[0][int(top_word_index.item() / 4)])

            previous_sequence = deepcopy(current_sequence)
        
        return current_sequence[0]
    '''