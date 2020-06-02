import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_pmf as conf 

class pmf(nn.Module):
    def __init__(self):
        super(pmf, self).__init__()
        self.user_embedding = nn.Embedding(conf.num_users, conf.mf_dimension)
        self.item_embedding = nn.Embedding(conf.num_items, conf.mf_dimension)

        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)

        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda()

        self.obj_function = nn.MSELoss(reduction='sum')
        self.loss_function = nn.MSELoss(reduction='none')

        self.reinit()

    def reinit(self):
        self.user_embedding.weight = torch.nn.Parameter(0.1 * self.user_embedding.weight)
        self.item_embedding.weight = torch.nn.Parameter(0.1 * self.item_embedding.weight)
        self.user_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_users, 1))
        self.item_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_items, 1))

    def forward(self, user, item, label):
        user_embed = self.embedding_user(user)
        item_embed = self.embedding_item(item)

        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
                
        output_embed = user_embed * item_embed
                
        prediction = torch.sum(output_embed, 1, keepdims=True) + self.avg_rating + user_bias + item_bias 

        obj_loss = self.obj_function(prediction.view(-1), label) 

        mse_loss = self.loss_function(prediction.view(-1), label)
        
        return prediction.view(-1), obj_loss, mse_loss