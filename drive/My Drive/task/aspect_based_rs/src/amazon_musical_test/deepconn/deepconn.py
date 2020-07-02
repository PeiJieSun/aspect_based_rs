import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_deepconn as conf


class encoder_deepconn(nn.Module):
    def __init__(self):
        super(encoder_deepconn, self).__init__()

        # parameters for DeepCoNN
        self.user_word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)  # vocab_size * 300
        self.item_word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dim)  # vocab_size * 300

        self.user_cnn = nn.Conv2d(1, conf.filters_num, (conf.kernel_size, conf.word_dim))
        self.item_cnn = nn.Conv2d(1, conf.filters_num, (conf.kernel_size, conf.word_dim))

        self.user_fc_linear = nn.Linear(conf.filters_num, conf.embedding_dim)
        self.item_fc_linear = nn.Linear(conf.filters_num, conf.embedding_dim)

        self.dropout = nn.Dropout(conf.dropout)

        self.reinit()

    def reinit(self):
        torch.manual_seed(0)
        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        torch.manual_seed(0)
        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        torch.manual_seed(0); nn.init.xavier_normal_(self.user_word_embedding.weight)
        torch.manual_seed(0); nn.init.xavier_normal_(self.item_word_embedding.weight)

    def forward(self, user_doc, item_doc):
        # Two CNN modules for user & item review-embedding representation        
        user_doc = self.user_word_embedding(user_doc)
        item_doc = self.item_word_embedding(item_doc)

        u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)

        u_fea = self.user_fc_linear(u_fea)
        i_fea = self.item_fc_linear(i_fea)
        
        u_out = self.dropout(u_fea)
        i_out = self.dropout(i_fea)

        return u_out, i_out

class decoder_fm(nn.Module):
    def __init__(self):
        super(decoder_fm, self).__init__()

        torch.manual_seed(0);
        
        self.free_user_embedding = nn.Embedding(conf.num_users, conf.embedding_dim)
        self.free_item_embedding = nn.Embedding(conf.num_items, conf.embedding_dim)

        self.dropout = nn.Dropout(conf.dropout)

        dim = conf.embedding_dim * 2
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(conf.num_users, 1))
        self.b_items = nn.Parameter(torch.randn(conf.num_items, 1))

        self.reinit()

    def reinit(self):
        torch.manual_seed(0); nn.init.xavier_normal_(self.free_user_embedding.weight)
        torch.manual_seed(0); nn.init.xavier_normal_(self.free_item_embedding.weight)

        torch.manual_seed(0);
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def forward(self, user, item, u_out, i_out):
        u_out = 0.0*u_out + self.free_user_embedding(user)
        i_out = 0.0*i_out + self.free_item_embedding(item)

        input_vec = torch.cat([u_out, i_out], 1)

        input_vec = self.dropout(input_vec)

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part + self.b_users[user] + self.b_items[item]  + conf.avg_rating

        prediction = fm_output.squeeze(1)

        return prediction

class deepconn(nn.Module):
    def __init__(self):
        super(deepconn, self).__init__()

        self.encoder = encoder_deepconn()
        self.decoder = decoder_fm()

        '''1_RATING PREDICTION ATTENTION PLEASE!!!'''
        #### START ------ ****** veriify rating prediction with PMF ****** ####
        #### FIRST PART #### 
        '''
        torch.manual_seed(0); self.embedding_user = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.embedding_item = nn.Embedding(conf.num_items, conf.mf_dim)
        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)
        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda() 
        
        self.reinit() ### 
        '''
        #### ****** veriify rating prediction with PMF ****** ------ END ####
        

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

    def forward(self, user, item, label, user_doc, item_doc):
        u_out, i_out = self.encoder(user_doc, item_doc)
        pred = self.decoder(user, item, u_out, i_out)
        

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


        rating_loss = F.mse_loss(pred, label, reduction='none')
        obj_loss = F.mse_loss(pred, label, reduction='sum')

        return obj_loss, rating_loss, pred