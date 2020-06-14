import torch
import torch.nn as nn
import torch.nn.functional as F 

from train_probe import conf

class probe(nn.Module):
    def __init__(self):
        super(probe, self).__init__()
        
        torch.manual_seed(0); self.embedding_user = nn.Embedding(conf.num_users, conf.mf_dim)
        torch.manual_seed(0); self.embedding_item = nn.Embedding(conf.num_items, conf.mf_dim)

        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.item_bias = nn.Embedding(conf.num_items, 1)

        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda()

        self.reinit()

    def reinit(self):
        self.embedding_user.weight = torch.nn.Parameter(0.1 * self.embedding_user.weight)
        self.embedding_item.weight = torch.nn.Parameter(0.1 * self.embedding_item.weight)
        self.user_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_users, 1))
        self.item_bias.weight = torch.nn.Parameter(torch.zeros(conf.num_items, 1))

    def forward(self, user, item, label, *argv):
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)

        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
                
        output_emb = user_emb * item_emb

        prediction = torch.sum(output_emb, 1, keepdims=True) + self.avg_rating + user_bias + item_bias

        obj_loss = F.mse_loss(prediction.view(-1), label, reduction='sum') 
        mse_loss = F.mse_loss(prediction.view(-1), label, reduction='none')

        #import pdb; pdb.set_trace()

        #return prediction.view(-1), obj_loss, mse_loss
        return mse_loss, obj_loss