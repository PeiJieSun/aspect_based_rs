import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_pmf as conf 

class pmf(nn.Module):
    def __init__(self):
        super(pmf, self).__init__()
        self.embedding_user = nn.Embedding(conf.num_users, conf.embedding_dim)
        self.embedding_item = nn.Embedding(conf.num_items, conf.embedding_dim)

        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)

        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda()

        self.obj_function = nn.MSELoss(reduction='sum')
        self.loss_function = nn.MSELoss(reduction='none')

        self.reinit()

    def reinit(self):
        self.embedding_user.weight = torch.nn.Parameter(0.1 * self.embedding_user.weight)
        self.embedding_item.weight = torch.nn.Parameter(0.1 * self.embedding_item.weight)
        self.user_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_users, 1))
        self.item_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_items, 1))

    def forward(self, user, item, label):
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)

        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
                
        output_emb = user_emb * item_emb
        
        #import pdb; pdb.set_trace()
        
        prediction = torch.sum(output_emb, 1, keepdims=True) + user_bias + item_bias + self.avg_rating

        obj_loss = self.obj_function(prediction.view(-1), label) 

        mse_loss = self.loss_function(prediction.view(-1), label)
        #rmse_loss = torch.sqrt(mse_loss)
        #import pdb; pdb.set_trace()
        
        return prediction.view(-1), obj_loss, mse_loss