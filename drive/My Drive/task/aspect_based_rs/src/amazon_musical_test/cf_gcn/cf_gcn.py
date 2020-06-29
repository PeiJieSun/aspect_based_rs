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

        return [
            gamma_user_embed, 
            gamma_item_embed, 
            theta_user_embed, 
            theta_item_embed, 
            user_doc_embed, 
            item_doc_embed
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
        self.rnn = nn.GRU(conf.word_dim+2*conf.mf_dim, conf.hidden_dim, num_layers=1)

        self.reinit()

    def reinit(self):
        torch.manual_seed(0); 
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, input_vector, hidden_state):
        input_vector = F.dropout(input_vector, p=0.5)

        output, hidden_state = self.rnn(input_vector, hidden_state)
        return output, hidden_state

class cf_gcn(nn.Module):
    def __init__(self):
        super(cf_gcn, self).__init__()

        self.encoder = encoder()
        self.decoder_rating = decoder_rating()
        self.decoder_review = decoder_review()

        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)
        self.rnn_out_linear = nn.Linear(conf.hidden_dim, conf.vocab_sz)
        
        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_uniform_(self.rnn_out_linear.weight)
        nn.init.zeros_(self.rnn_out_linear.bias)
    
    # values: user, item, label, review_input, review_target, user_doc, item_doc
    def forward(self, values):
        [user, item, label, review_input, review_target, user_doc, item_doc] = values

        embed = self.encoder(user, item, user_doc, item_doc)

        ###### review generation
        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda()
        outputs = []
        for t_input in review_input:
            input_vector = self.word_embedding(t_input)
            input_vector = torch.cat([input_vector, embed[0], embed[1]], dim=1).view(1, user.shape[0], -1)

            output, hidden_state = self.decoder_review(input_vector, hidden_state)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).view(-1, conf.hidden_dim) # (tiem*batch, hidden_size)

        word_probit = self.rnn_out_linear(outputs) # (time*batch, vocab_sz)
        
        review_out_loss = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD, reduction='none')
        review_obj = F.cross_entropy(word_probit, review_target.reshape(-1), ignore_index=PAD)

        ###### rating prediction
        pred = self.decoder_rating(user, item, embed)

        rating_out_loss = F.mse_loss(pred, label, reduction='none')
        rating_obj = F.mse_loss(pred, label, reduction='sum')

        obj = 1*rating_obj + 1e-10*review_obj

        return review_out_loss, rating_out_loss, obj

    def predict_rating(self, values):
        [user, item, label, review_input, review_target, user_doc, item_doc] = values

        embed = self.encoder(user, item, user_doc, item_doc)

        ###### rating prediction
        pred = self.decoder_rating(user, item, embed)
        rating_out_loss = F.mse_loss(pred, label, reduction='none')

        return rating_out_loss

    def _sample_text_by_top_one(self, user, item, review_input, user_doc, item_doc):
        embed = self.encoder(user, item, user_doc, item_doc)
        
        hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).cuda()
        next_word_idx = review_input[0]

        sample_idx_list = [next_word_idx]
        for _ in range(conf.rev_len):
            input_vector = self.word_embedding(next_word_idx)
            input_vector = torch.cat([input_vector, embed[0], embed[1]], dim=1).view(1, user.shape[0], -1)

            output, hidden_state = self.decoder_review(input_vector, hidden_state)
            word_probit = self.rnn_out_linear(output).reshape(-1, conf.vocab_sz)

            next_word_idx = torch.argmax(word_probit, 1)
                
            sample_idx_list.append(next_word_idx)

        sample_idx_list = torch.stack(sample_idx_list, dim=0).transpose(0, 1)
        return sample_idx_list