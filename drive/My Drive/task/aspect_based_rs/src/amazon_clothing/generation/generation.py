import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_generation as conf

class generation(nn.Module):
    def __init__(self):
        super(generation, self).__init__()

        # parameters for aspect extraction
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension) 
        self.word_embedding.weight.requires_grad = False
        
        self.transform_M = nn.Linear(conf.word_dimension, conf.word_dimension, bias=False) # weight: word_dimension * word_dimension
        self.transform_W = nn.Linear(conf.word_dimension, conf.aspect_dimension) # weight: aspect_dimension * word_diension
        self.transform_T = nn.Linear(conf.aspect_dimension, conf.word_dimension, bias=False) # weight: word_dimension * aspect_dimension

        self.transform_wang = nn.Linear(conf.aspect_dimension, conf.aspect_dimension)

        self.free_user_embedding = nn.Embedding(conf.num_users, conf.aspect_dimension)
        self.free_item_embedding = nn.Embedding(conf.num_items, conf.aspect_dimension)

        self.user_fc_linear = nn.Linear(2*conf.aspect_dimension, 2*conf.embedding_dim)
        dim = conf.embedding_dim * 2
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(conf.num_users, 1))
        self.b_items = nn.Parameter(torch.randn(conf.num_items, 1))

        self.dropout = nn.Dropout(conf.drop_out)

        self.mse_func_1 = nn.MSELoss(reduction='none')
        self.mse_func_2 = nn.MSELoss()
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=1.0, reduction='none')

        self.reset_para()

        # PARAMETERS FOR LSTM
        self.rnn = nn.GRU(conf.word_dimension, conf.hidden_size, num_layers=1)

        #self.gamma_user_embedding = nn.Embedding(conf.num_users, conf.m)
        self.gamma_item_embedding = nn.Embedding(conf.num_items, conf.m)

        self.beta_user_embedding = nn.Embedding(conf.num_users, conf.k)
        self.beta_item_embedding = nn.Embedding(conf.num_items, conf.k)

        self.u_linear = nn.Linear(2*conf.m, conf.n)
        self.v_linear = nn.Linear(2*conf.k, conf.n)

        self.linear_1 = nn.Linear(conf.m + conf.n, 1)
        self.linear_2 = nn.Linear(2*conf.k, conf.k)
        self.linear_3 = nn.Linear(conf.k+conf.word_dimension+conf.n, conf.k)
        self.linear_5 = nn.Linear(conf.n+conf.m, conf.vocab_sz)

    def reset_para(self):
        nn.init.uniform_(self.free_user_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.free_item_embedding.weight, a=-0.1, b=0.1)

        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)
    
    def predict_rating(self, pos_review, # (num_review, sequence_length)
        neg_review, # (num_review * num_negative, word_dimension)
        user, item, label, # (batch_size, 1)
        user_idx_list, item_idx_list):
        ########################### First: encode the historical review, and get the aspect-based review embedding ###########################
        # encode the historical reviews of the target user and item(mapping the historical reviews of the target user and item to the aspect space)
        # historical_review_set ==> aspect-based historical review embedding
        w = pos_review; 

        positive_review_embed = self.word_embedding(w) # (batch_size, sequence_length, word_dimension)
        negative_review_embed = self.word_embedding(neg_review) # (batch_size*num_negative_reviews, sequence_length, word_dimension)

        y_s = torch.mean(positive_review_embed, 1).view(-1, conf.word_dimension) # (batch_size, word_dimension)
        z_n = torch.mean(negative_review_embed, 1).view(-1, conf.word_dimension) # (batch_size * num_negative_reviews, word_dimension)

        e_w = self.word_embedding(w) # (num_review, sequence_length, word_dimension)
        y_s = y_s.view(y_s.shape[0], y_s.shape[1], 1) # (num_review, word_dimension, 1)
        
        # self.trainsofmr_M(e_w): (num_review, sequence_length, word_dimension)
        dx = torch.matmul(self.transform_M(e_w), y_s) # (num_review, sequence_length, 1)
        ax = F.softmax(dx, dim=1) # (num_review, sequence_length, 1)     
        
        # e_w.view(e_w.shape[0], e_w.shape[2], -1): (num_review, word_dimension, sequence_length)
        # torch.matmul(e_w, a): (num_review, word_dimension, 1)
        z_s = torch.matmul(e_w.view(e_w.shape[0], e_w.shape[2], -1), ax).view(-1, conf.word_dimension) # (num_review, word_dimension)

        # self.transform_W(z_s): (num_review, aspect_dimension)
        #p_t = F.softmax(self.transform_W(z_s), dim=1) # (num_review, aspect_dimension)
        p_t = self.transform_W(z_s)
        r_s = self.transform_T(p_t) # (num_review, word_dimension)
        
        # cosine similarity betwee r_s and z_s
        c1 = (F.normalize(r_s, p=2, dim=1) * F.normalize(z_s, p=2, dim=1)).sum(-1, keepdim=True) # (num_review, 1)
        c1 = c1.repeat(1, conf.num_negative_reviews).view(-1) # (num_review * num_negative)

        # z_n.view(conf.num_review, conf.num_negative_reviews, -1): (num_review, num_negative_reviews, word_dimension)
        # r_s.view(conf.num_review, 1, -1): (num_review, 1, word_dimension)
        # z_n * r_s: (num_review, num_negative_reviews, word_dimension)
        # (z_n * r_s).sum(-1): (num_review, num_negative)
        # (z_n * r_s).sum(-1).sum(-1): (num_review)
        c2 = (F.normalize(z_n.view(y_s.shape[0], conf.num_negative_reviews, -1), p=2, dim=2) \
             * F.normalize(r_s.view(y_s.shape[0], 1, -1), p=2, dim=2)).sum(-1).view(-1) # (num_review * num_negative)

        abae_out_loss = self.margin_ranking_loss(c1, c2, torch.FloatTensor([1.0]).cuda())

        J_loss = torch.mean(abae_out_loss)

        transform_T_weight = F.normalize(self.transform_T.weight, p=2, dim=0) # word_dimension * aspect_dimension
        U_loss = self.mse_func_2(torch.matmul(torch.transpose(transform_T_weight, 0, 1), transform_T_weight), torch.eye(conf.aspect_dimension).cuda())
        
        ########################### Second: collect the aspect-based user embedding and item embedding ###########################
        user_aspect_embed = p_t[user_idx_list].view(label.shape[0], -1, conf.aspect_dimension) # (batch_size, u_max_r, mf_dimension)
        item_aspect_embed = p_t[item_idx_list].view(label.shape[0], -1, conf.aspect_dimension) # (batch_size, i_max_r, mf_dimension)

        user_aspect_embed = torch.mean(user_aspect_embed, 1) # (batch_size, xx_dimension)
        item_aspect_embed = torch.mean(item_aspect_embed, 1) # (batch_size, xx_dimension)

        item_aspect_embed = self.transform_wang(item_aspect_embed)

        # Co-Attention
        user_aspect_embed = torch.sum(torch.matmul(user_aspect_embed.view(label.shape[0], -1, 1), \
            item_aspect_embed.view(label.shape[0], 1, -1)), 1)

        #item_aspect_embed = torch.sum(torch.matmul(user_aspect_embed.view(label.shape[0], -1, 1), \
        #    item_aspect_embed.view(label.shape[0], 1, -1)), 2)
        
        self.aspect_user_embed = user_aspect_embed
        self.aspect_item_embed = item_aspect_embed

        x_user_aspect_embed = user_aspect_embed + self.free_user_embedding(user)
        x_item_aspect_embed = item_aspect_embed + self.free_item_embedding(item)

        input_vec = torch.cat([x_user_aspect_embed, x_item_aspect_embed], 1) # (batch_size, 2*xx_dimension)
        input_vec = self.user_fc_linear(input_vec)
        input_vec = self.dropout(input_vec)

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part + self.b_users[user] + self.b_items[item] # + conf.avg_rating

        prediction = fm_output.squeeze(1)
        
        rating_loss = self.mse_func_1(prediction, label)
        mse_loss = self.mse_func_2(prediction, label)

        # collect the loss of abae and rating prediction
        obj_loss = mse_loss + 0.001*J_loss + 0.001*U_loss

        return obj_loss, rating_loss, abae_out_loss, prediction

    def generate_review(self, user, item, label, review_input, review_output, \
        review_aspect, review_aspect_bool):
        ########################### FIRST: GET THE ASPECT-BASED REVIEW EMBEDDING ##########################
        gamma_u = self.free_user_embedding(user) # (batch_size, m)
        gamma_i = self.free_item_embedding(item) # (batch_size, m)

        beta_u = self.beta_user_embedding(user) # (batch_size, k)
        beta_i = self.beta_item_embedding(item) # (batch_size, k)

        #beta_u = self.aspect_user_embed
        #beta_i = self.aspect_item_embed

        u_vector = torch.tanh(self.u_linear(torch.cat([gamma_u, gamma_i], 1))) # (batch_size, n)
        v_vector = torch.tanh(self.v_linear(torch.cat([beta_u, beta_i], 1))) # (batch_size, n)

        h_0 = (u_vector + v_vector).view(1, user.shape[0], conf.hidden_size) # (1 * 1, batch_size, hidden_size=n)

        review_input_embed = self.word_embedding(review_input)# (seq_length, batch_size, word_dimension)

        outputs, h_n = self.rnn(review_input_embed, h_0) # (seq_length, batch_size, hidden_size=n)
        review_output_embed = outputs.view(-1, outputs.size()[2])#(seq_length * batch_size, hidden_size=n)
        
        # calculate a2t
        # gamma_u.repeat(outputs.shape[0], 1): (seq_length*batch_size, m)
        # torch.cat([gamma_u, review_output_embed], 1): (seq_length*batch_size, m+n)
        alpha_tu = torch.tanh(self.linear_1(torch.cat([gamma_u.repeat(outputs.shape[0], 1), review_output_embed], 1))) # (seq_length * batch_size, 1)
        
        alpha_ti = torch.tanh(self.linear_1(torch.cat([gamma_i.repeat(outputs.shape[0], 1), review_output_embed], 1))) # (seq_length * batch_size, 1)

        alpha = torch.cat([alpha_tu, alpha_ti], 1) # (seq_length * batch_size, 2)
        alpha = F.softmax(alpha, 1) # (seq_length * batch_size, 2)

        alpha = torch.transpose(alpha, 0, 1) # (2, seq_length * batch_size)
        alpha_tu, alpha_ti = alpha[0].view(-1, 1), alpha[1].view(-1, 1) # (seq_length * batch_size, 1)

        # gamma_u.view(1, user.shape[0], -1): (1, batch_size, m)
        # gamma_i.view(1, user.shape[0], -1): (1, batch_size, m)
        a2t = alpha_tu * gamma_u.repeat(outputs.shape[0], 1) + alpha_ti * gamma_i.repeat(outputs.shape[0], 1) # (seq_length * batch_size, m)
        #import pdb; pdb.set_trace()

        # calculate a3t
        # torch.cat([beta_u, beta_i], 1): (batch_size, 2*k)
        sui = self.linear_2(torch.cat([beta_u, beta_i], 1)) # (batch_size, k)

        # sui.repeat(outputs.shape[0]): (seq_length*batch_size, k)
        # torch.cat([sui.repeat(outputs.shape[0]), review_input_embed, review_output_embed], 1): (seq_length*batch_size, k+word_dim+n)
        a3t = torch.tanh(self.linear_3(torch.cat((sui.repeat(outputs.shape[0], 1), review_input_embed.view(-1, conf.word_dimension), review_output_embed), 1))) # (seq_length*batch_size, k)

        ############################### Pv(Wt) #########################################
        PvWt = torch.tanh(self.linear_5(torch.cat([review_output_embed, a2t], 1))) # (seq_length*batch_size, vocab_sz)

        ############################### P(Wt) #########################################
        aspect_probit = torch.index_select(a3t, 1, review_aspect) * review_aspect_bool # (seq_length*batch_size, vocab_sz)
        #aspect_probit = F.log_softmax(aspect_probit, 1)

        #PvWt = torch.tanh(self.linear_6(review_output_embed))
        Pwt = PvWt + aspect_probit
        obj_loss = F.nll_loss(F.log_softmax(Pwt, 1), review_output.view(-1), reduction='mean')

        return obj_loss

    def forward(self, pos_review, # (num_review, sequence_length)
        neg_review, # (num_review * num_negative, word_dimension)
        user, item, label, # (batch_size, 1)
        user_idx_list, item_idx_list, review_input,
        review_output, review_aspect, review_aspect_bool):
        
        obj_loss, rating_loss, abae_out_loss, prediction = \
            self.predict_rating(pos_review, neg_review, user, item, label, user_idx_list, item_idx_list)

        generation_loss = self.generate_review(user, item, label, \
            review_input, review_output, review_aspect, review_aspect_bool)

        obj_loss = obj_loss + 0.9*generation_loss
        return obj_loss, rating_loss, abae_out_loss, prediction, generation_loss

        '''
        obj_loss = self.generate_review(user, item, label, review_input, review_output, review_aspect, review_aspect_bool)
        return obj_loss
        '''
        