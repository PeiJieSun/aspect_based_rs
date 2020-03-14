import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_aspect_rs as conf 

class aspect_rs(nn.Module):
    def __init__(self):
        super(aspect_rs, self).__init__()
        self.embedding_dim = conf.embedding_dim
        self.num_user = conf.num_users
        self.num_item = conf.num_items

        torch.manual_seed(0); self.embedding_user = nn.Embedding(self.num_user, self.embedding_dim)
        torch.manual_seed(0); self.embedding_item = nn.Embedding(self.num_item, self.embedding_dim)

        self.rating_loss_function = nn.MSELoss()
        self.review_loss_function = nn.AdaptiveLogSoftmaxWithLoss(\
            conf.hidden_size, conf.vocab_sz, cutoffs=[round(conf.vocab_sz/15), 3*round(conf.vocab_sz/15)], div_value=2)

        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda()
    
    # user: batch user list
    # item: batch item list
    # label: batch label list
    # user_histor: batch user historical review id list, OrderDict
    # item_histor: batch item historical review id list, OrderDict
    # review_input: batch review input
    # review_output: batch review output
    # historical_review: 
    # historical_review_negative: 
    # historical_review_positive: 
    def forward(self, user, item, user_histor_idx, item_histor_idx, historical_review, historical_review_positive, historical_review_negative, review_input, label, review_output):        
        ########################### First: encode the historical review, and get the aspect-based review embedding ###########################
        # encode the historical reviews of the target user and item(mapping the historical reviews of the target user and 
        # item to the aspect space)
        # historical_review_set ==> aspect-based historical review embedding
        w = historical_review; y_s = historical_review_positive; z_n = historical_review_negative

        e_w = self.word_embedding(w) # (batch_size, sequence_length, word_dimension)
        y_s = y_s.view(y_s.shape[0], y_s.shape[1], 1) # (batch_size, word_dimension, 1)
        
        # self.trainsofmr_M(e_w): (batch_size, sequence_length, word_dimension)
        dx = torch.matmul(self.transform_M(e_w), y_s) # (batch_size, sequence_length, 1)
        ax = F.softmax(dx, dim=1) # (batch_size, sequence_length, 1)     
        
        # e_w.view(e_w.shape[0], e_w.shape[2], -1): (batch_size, word_dimension, sequence_length)
        # torch.matmul(e_w, a): (batch_size, word_dimension, 1)
        z_s = torch.matmul(e_w.view(e_w.shape[0], e_w.shape[2], -1), ax).view(-1, conf.word_dimension) # (batch_size, word_dimension)

        # self.transform_W(z_s): (batch_size, aspect_dimension)
        p_t = F.softmax(self.transform_W(z_s), dim=1) # (batch_size, aspect_dimension)
        r_s = self.transform_T(p_t) # (batch_size, word_dimension)

        aspect_based_review_embedding = []

        ########################### Second: collect the aspect-based user embedding and item embedding ###########################
        # get the embedding of the user and the item
        user_aspect_embedding, item_aspect_embedding = [], []
        for batch_idx, histor_idx_list in user_histor_idx.items():
            tmp_user_aspect_embedding = []
            for idx in histor_idx_list:
                tmp_user_aspect_embedding.append(aspect_based_review_embedding[idx])
            user_aspect_embedding.append(torch.mean(tmp_user_aspect_embedding, dim=xx))
        
        for batch_idx, histor_idx_list in item_histor_idx.items():
            tmp_item_aspect_embedding = []
            for idx in histor_idx_list:
                tmp_item_aspect_embedding.append(aspect_based_review_embedding[idx])
            item_aspect_embedding.append(torch.mean(tmp_item_aspect_embedding, dim=xx))

        user_aspect_embedding = torch.stack(user_aspect_embedding)
        item_aspect_embedding = torch.stack(item_aspect_embedding)

        ########################### Third: calculate the predicted rating ###########################
        # decode the aspect-based user and item embedding to the rating
        # rating = Decode(aspect-based user & item embedding) + Decode(matrix factorization)
        rating = user_aspect_embedding * item_aspect_embedding \
            + user_free_embedding * item_free_embedding \
            + avg_rating

        rating_loss = self.rating_loss_function(prediction, label)

        ########################### Fourth: generate the review ###########################
        # decode the aspect-based user and item embedding to the review
        review_input_embedding = self.word_embedding(review_input) #size: (sequence_length * batch_size) * self.conf.text_word_dimension
        outputs, hidden_state = self.rnn(review_input_embedding, hidden_state) # sequence_length * batch_size * hidden_dimension
        review_output_embedding = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_dimension]

        review_loss = self.review_loss_function()

        ########################### Fifth: collect the loss and return the key information ###########################
        
        return prediction.view(-1), loss