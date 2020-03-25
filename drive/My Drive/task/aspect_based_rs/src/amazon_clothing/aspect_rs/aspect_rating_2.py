import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_aspect as conf 

class aspect_rating_2(nn.Module):
    def __init__(self):
        super(aspect_rating_2, self).__init__()

        # parameters for aspect extraction
        self.word_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension) 
        self.word_embedding.weight.requires_grad = False
        
        self.transform_M = nn.Linear(conf.word_dimension, conf.word_dimension, bias=False) # weight: word_dimension * word_dimension
        self.transform_W = nn.Linear(conf.word_dimension, conf.aspect_dimension) # weight: aspect_dimension * word_diension
        self.transform_T = nn.Linear(conf.aspect_dimension, conf.word_dimension, bias=False) # weight: word_dimension * aspect_dimension

    # user/item/label: batch user/item/label list
    # user_histor/item_histor: batch user/item historical review id list, OrderDict
    # historical_review: 
    # review_positive/review_negative: used to train the abae model
    def forward(self, historical_review, # (num_review, sequence_length)
        review_positive, # (num_review, word_dimension)
        review_negative, # (num_review * num_negative, word_dimension)
        user_histor_index, user_histor_value, item_histor_index, item_histor_value):
        ########################### First: encode the historical review, and get the aspect-based review embedding ###########################
        # encode the historical reviews of the target user and item(mapping the historical reviews of the target user and 
        # item to the aspect space)
        # historical_review_set ==> aspect-based historical review embedding
        w = historical_review; y_s = review_positive; z_n = review_negative
        e_w = self.word_embedding(w) # (num_review, sequence_length, word_dimension)
        y_s = y_s.view(y_s.shape[0], y_s.shape[1], 1) # (num_review, word_dimension, 1)
        
        #import pdb; pdb.set_trace()

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
        
        ########################### Second: collect the aspect-based user embedding and item embedding ###########################
        # get the embedding of the user and the item
        user_histor_tensor = \
            torch.sparse.FloatTensor(user_histor_index, user_histor_value, torch.Size([label.shape[0], w.shape[0]])) # (batch_size, num_review)
        item_histor_tensor = \
            torch.sparse.FloatTensor(item_histor_index, item_histor_value, torch.Size([label.shape[0], w.shape[0]])) # (batch_size, num_review)

        user_aspect_embedding = torch.sparse.mm(user_histor_tensor, p_t) # (batch_size, mf_dimension)
        item_aspect_embedding = torch.sparse.mm(item_histor_tensor, p_t) # (batch_size, mf_dimension)
        
        return user_aspect_embedding, item_aspect_embedding

class aspect_rating_3(nn.Module):
    def __init__(self):
        super(aspect_rating_3, self).__init__()

        # parameters for aspect extraction
        self.user_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension) 
        self.item_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension) 

        self.user_aspect_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension) 
        self.item_aspect_embedding = nn.Embedding(conf.vocab_sz, conf.word_dimension) 
        
        # parameters for rating prediction
        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda()

        # loss function
        self.mse_loss = nn.MSELoss(reduction='none')
        self.abae_mse_loss = nn.MSELoss(reduction='mean')
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=1.0, reduction='none')

        # FM parameters
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(conf.aspect_dimension, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(conf.aspect_dimension, 10))
        self.b_users = nn.Parameter(torch.randn(conf.num_users, 1))
        self.b_items = nn.Parameter(torch.randn(conf.num_items, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)
    
    # user/item/label: batch user/item/label list
    # user_histor/item_histor: batch user/item historical review id list, OrderDict
    # historical_review: 
    # review_positive/review_negative: used to train the abae model
    def forward(self, user, item, label): # (batch_size, 1)
        ########################### First: encode the historical review, and get the aspect-based review embedding ###########################
        # encode the historical reviews of the target user and item(mapping the historical reviews of the target user and item to the aspect space)
        
        #import pdb; pdb.set_trace()

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
        U_loss = self.abae_mse_loss(torch.matmul(torch.transpose(transform_T_weight, 0, 1), transform_T_weight), torch.eye(conf.aspect_dimension).cuda())

        abae_loss = U_loss + J_loss
        ########################### Second: collect the aspect-based user embedding and item embedding ###########################
        # get the embedding of the user and the item
        
        user_histor_tensor = \
            torch.sparse.FloatTensor(user_histor_index, user_histor_value, torch.Size([label.shape[0], w.shape[0]])) # (batch_size, num_review)
        item_histor_tensor = \
            torch.sparse.FloatTensor(item_histor_index, item_histor_value, torch.Size([label.shape[0], w.shape[0]])) # (batch_size, num_review)

        user_aspect_embedding = torch.sparse.mm(user_histor_tensor, p_t) # (batch_size, mf_dimension)
        item_aspect_embedding = torch.sparse.mm(item_histor_tensor, p_t) # (batch_size, mf_dimension)
        
        '''
        user_histor_tensor = torch.sparse.FloatTensor(user_histor_index, user_histor_value, \
            torch.Size([label.shape[0], w.shape[0]])).to_dense() # (batch_size, num_review)
        item_histor_tensor = torch.sparse.FloatTensor(item_histor_index, item_histor_value, \
            torch.Size([label.shape[0], w.shape[0]])).to_dense() # (batch_size, num_review)

        user_aspect_embedding = torch.mm(user_histor_tensor, r_s) # (batch_size, mf_dimension)
        item_aspect_embedding = torch.mm(item_histor_tensor, r_s) # (batch_size, mf_dimension)
        '''

        ########################### Third: calculate the predicted rating ###########################
        # decode the aspect-based user and item embedding to the rating
        # rating = Decode(aspect-based user & item embedding) + Decode(matrix factorization)
        
        output_emb = user_aspect_embedding * item_aspect_embedding # (batch_size, mf_dimension)

        #prediction = output_emb.sum(-1, keepdims=True) + self.avg_rating # (batch_size, 1)

        '''
        y = w_0 + \sum {w_ix_i} + \sum_{i=1}\sum_{j=i+1}<v_i, v_j>x_ix_j
        factorization machine layer
        refer: https://github.com/vanzytay/KDD2018_MPCN/blob/master/tylib/lib
                      /compose_op.py#L13
        '''

        input_vec = output_emb

        # linear part: first two items
        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2) + fm_linear_part

        prediction = fm_output + self.avg_rating #+ self.b_users[uids] + self.b_items[iids]

        #import pdb; pdb.set_trace()

        rating_out_loss = self.mse_loss(prediction.view(-1), label)
        rating_loss = torch.mean(rating_out_loss)

        ########################### Fourth: collect the loss and return the key information ###########################
        obj = conf.lr_rating * rating_loss + conf.lr_abae * abae_loss

        #import pdb; pdb.set_trace()
        return obj, rating_out_loss, abae_out_loss, prediction, user_aspect_embedding, item_aspect_embedding