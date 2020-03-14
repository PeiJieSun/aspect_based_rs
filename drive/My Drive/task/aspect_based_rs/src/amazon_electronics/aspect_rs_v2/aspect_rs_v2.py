import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_aspect_rs_v1 as conf 

class aspect_rs_v1(nn.Module):
    def __init__(self):
        super(aspect_rs_v1, self).__init__()
        self.embedding_dim = conf.embedding_dim
        self.num_user = conf.num_users
        self.num_item = conf.num_items

        torch.manual_seed(0); self.embedding_user = nn.Embedding(self.num_user, self.embedding_dim)
        torch.manual_seed(0); self.embedding_item = nn.Embedding(self.num_item, self.embedding_dim)

        self.rating_loss_function = nn.MSELoss()
        self.review_loss_function = nn.AdaptiveLogSoftmaxWithLoss(\
            conf.hidden_size, conf.vocab_sz, cutoffs=[round(conf.vocab_sz/15), 3*round(conf.vocab_sz/15)], div_value=2)
    
    def forward(self, user, item, review_input, label, review_output):        
        ########################### First: get the implicit aspect-based review embedding ###########################


        ########################### Second: calculate the predicted rating ###########################
        # decode the aspect-based user and item embedding to the rating
        # rating = Decode(aspect-based user & item embedding) + Decode(matrix factorization)
        rating = user_aspect_embedding * item_aspect_embedding \
            + user_free_embedding * item_free_embedding \
            + avg_rating

        rating_loss = self.rating_loss_function(prediction, label)

        ########################### Third: generate the review ###########################
        # decode the aspect-based user and item embedding to the review
        review_input_embedding = self.word_embedding(review_input) #size: (sequence_length * batch_size) * self.conf.text_word_dimension
        outputs, hidden_state = self.rnn(review_input_embedding, hidden_state) # sequence_length * batch_size * hidden_dimension
        review_output_embedding = outputs.view(-1, outputs.size()[2])#[sequence_length * batch_size, hidden_dimension]

        review_loss = self.review_loss_function()

        ########################### Fourth: collect the loss and return the key information ###########################
        
        return prediction.view(-1), loss