import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_pmf as conf 

class pmf(nn.Module):
    def __init__(self):
        super(pmf, self).__init__()
        torch.manual_seed(0); self.embedding_user = nn.Embedding(conf.num_users, conf.embedding_dim)
        torch.manual_seed(0); self.embedding_item = nn.Embedding(conf.num_items, conf.embedding_dim)

        self.avg_rating = torch.FloatTensor([conf.avg_rating]).cuda()

        self.obj_function = nn.MSELoss()
        self.loss_function = nn.MSELoss(reduction='none')

    def forward(self, user, item, label):
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
                
        output_emb = user_emb * item_emb
        
        #import pdb; pdb.set_trace()
        
        prediction = torch.sum(output_emb, 1, keepdims=True) + self.avg_rating

        obj_loss = self.obj_function(prediction.view(-1), label)

        mse_loss = self.loss_function(prediction.view(-1), label)
        rmse_loss = torch.sqrt(mse_loss)
        #import pdb; pdb.set_trace()
        
        return prediction.view(-1), obj_loss, rmse_loss