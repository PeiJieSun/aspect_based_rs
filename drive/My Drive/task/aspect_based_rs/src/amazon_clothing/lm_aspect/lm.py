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

import config_lm as conf 

class lm(nn.Module): 
    def __init__(self):
        super(lm, self).__init__()
        
        self.user_embedding = nn.Embedding(conf.num_users, conf.mf_dimension)
        self.item_embedding = nn.Embedding(conf.num_items, conf.mf_dimension)

        self.user_linear = nn.Linear(conf.mf_dimension, conf.hidden_dimension)
        self.item_linear = nn.Linear(conf.mf_dimension, conf.hidden_dimension)

        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension)
        self.word_embedding.weight.requires_grad = False
        
        self.rnn = nn.GRU(conf.word_dimension, conf.hidden_dimension, num_layers=1, bidirectional=False)

        self.rnn_out_linear = nn.Linear(conf.hidden_dimension, conf.vocab_sz)
        
    def forward(self, user, item, review_input, review_target):
        user_vector = self.user_linear(self.user_embedding(user))
        item_vector = self.item_linear(self.item_embedding(item))
        cat_vector = user_vector + item_vector

        hidden_state = torch.reshape(cat_vector, (1, -1, conf.hidden_dimension))

        word_vector = self.word_embedding(review_input) #size: (time*batch) * self.conf.text_word_dimension
        outputs, _ = self.rnn(word_vector, hidden_state) # time * batch * hidden_dimension
        
        word_probit = self.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz) # time * batch * vocab_sz


        word_probit = F.log_softmax(word_probit, dim=1) 
        out_loss = F.nll_loss(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        obj = F.nll_loss(word_probit, review_target.reshape(-1), ignore_index=PAD)
        
        #out_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        #obj = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD)

        return out_loss, obj
    
    def sampleTextByTemperature(self, user, item):
        temperature = 0.7

        user_vector = self.user_linear(self.user_embedding(user))
        item_vector = self.item_linear(self.item_embedding(item))
        cat_vector = user_vector + item_vector

        hidden_state = torch.reshape(cat_vector, (1, 1, conf.hidden_dimension))

        sample_idx_list = [SOS]
        next_word_idx = torch.LongTensor([SOS]).cuda()
        for _ in range(conf.sequence_length):
            seed_vector = self.word_embedding(next_word_idx).reshape(1, 1, -1)

            outputs, hidden_state = self.rnn(seed_vector, hidden_state)
            word_probit = self.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)
            
            probit = word_probit.view(-1).div(temperature)
            probit = torch.clamp(probit, max=80)
            probit = probit.exp()

            next_word_idx = torch.multinomial(probit, 1)[0]
            if next_word_idx.item() == PAD:
                return sample_idx_list

            sample_idx_list.append(next_word_idx.item())
        return sample_idx_list
    
    def sampleTextByTopOne(self, user, item):
        user_vector = self.user_linear(self.user_embedding(user))
        item_vector = self.item_linear(self.item_embedding(item))
        cat_vector = user_vector + item_vector

        hidden_state = torch.reshape(cat_vector, (1, 1, conf.hidden_dimension))

        sample_idx_list = [SOS]
        next_word_idx = torch.LongTensor([SOS]).cuda()
        for _ in range(conf.sequence_length):
            seed_vector = self.word_embedding(next_word_idx).reshape(1, 1, -1)

            outputs, hidden_state = self.rnn(seed_vector, hidden_state)
            word_probit = self.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)

            next_word_idx = torch.argmax(word_probit, 1)
            #import pdb; pdb.set_trace()
            if next_word_idx.item() == PAD:
                return sample_idx_list
                
            sample_idx_list.append(next_word_idx.item())
        return sample_idx_list

    def sampleTextByBeamSearch(self, user, item):
        beam_size = 4

        previous_sequence = {}
        current_sequence = {}

        # initialize the dictionary which stores the generated words token idx
        for sub_idx in range(beam_size):
            previous_sequence[sub_idx] = [SOS]

        # first iteration: prepare the input text data
        next_word_idx = torch.LongTensor([SOS]).cuda()
        seed_vector = self.word_embedding(next_word_idx).reshape(1, 1, -1)

        user_vector = self.user_linear(self.user_embedding(user))
        item_vector = self.item_linear(self.item_embedding(item))
        cat_vector = user_vector + item_vector

        hidden_state = torch.reshape(cat_vector, (1, 1, conf.hidden_dimension))
    
        outputs, hidden_state = self.rnn(seed_vector, hidden_state) # output_word_embedding: (1*batch_size, word_dimension)
        word_probit = self.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)

        # values: (batch_size, beam_size)
        values, indices = torch.topk(word_probit, beam_size) # indices: (batch_size, beam_size)

        # prepare the input data for next iteration
        next_word_idx = indices.view(1, -1) # (1, batch_size*beam_size)
        previous_probit = values.view(-1, 1) # (batch_size*beam_size, 1)

        for sub_idx in range(next_word_idx.shape[1]):
            previous_sequence[sub_idx].append(indices[0][sub_idx].item())

        #second iteration: construct the hidden_state for current iteration
        hidden_state = hidden_state.repeat(1, beam_size, 1)  # (1, beam_size, hidden_size)

        for _ in range(conf.sequence_length):
            seed_vector = self.word_embedding(next_word_idx)

            # hidden_state: ((2, batch_size, hidden_size), (2, batch_size, hidden_size))
            outputs, hidden_state = self.rnn(seed_vector, hidden_state) # review_output_embedding: (batch_size*beam_size, word_dimension)
            word_probit = self.rnn_out_linear(outputs).reshape(-1, conf.vocab_sz)

            tmp_hidden_state = deepcopy(hidden_state.data)

            current_probit = word_probit + previous_probit # (batch_size*beam_size, vocab_size)
            
            first_values, first_indices = torch.topk(current_probit, beam_size) # (batch_size*beam_size, beam_size)

            first_values = first_values.view(-1) # (batch_size, beam_size*beam_size)
            first_indices = first_indices.view(-1) # (batch_size, beam_size*beam_size)

            # second_values: (batch_size, beam_size)
            second_values, second_indices = torch.topk(first_values, beam_size) # second_indices: (batch_size, beam_size)

            #import pdb; pdb.set_trace()

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