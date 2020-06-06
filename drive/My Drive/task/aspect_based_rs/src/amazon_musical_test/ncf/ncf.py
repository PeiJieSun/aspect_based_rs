import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_ncf as conf 

class ncf(nn.Module):
    def __init__(self):
        super(ncf, self).__init__()
        self.gmf_user_embedding = nn.Embedding(conf.num_users, conf.gmf_embed_dim)
        self.gmf_item_embedding = nn.Embedding(conf.num_items, conf.gmf_embed_dim)

        self.mlp_user_embedding = nn.Embedding(conf.num_users, conf.mlp_embed_dim)
        self.mlp_item_embedding = nn.Embedding(conf.num_items, conf.mlp_embed_dim)

        self.linears = []
        for idx in range(1, len(conf.mlp_dim_list)):
            self.linears.append(nn.Linear(conf.mlp_dim_list[idx-1], conf.mlp_dim_list[idx], bias=False).cuda())

        self.final_linear = nn.Linear(conf.mlp_embed_dim+conf.gmf_embed_dim, 1)

        self.obj_function = nn.MSELoss(reduction='sum')
        self.loss_function = nn.MSELoss(reduction='none')

        self.reinit()

    def reinit(self):
        nn.init.xavier_normal_(self.gmf_user_embedding.weight)
        nn.init.xavier_normal_(self.gmf_item_embedding.weight)

        nn.init.xavier_normal_(self.mlp_user_embedding.weight)
        nn.init.xavier_normal_(self.mlp_item_embedding.weight)

        for idx in range(len(conf.mlp_dim_list)-1):
            nn.init.uniform_(self.linears[idx].weight, -0.1, 0.1)
    
        nn.init.uniform_(self.final_linear.weight, -0.05, 0.05)
        nn.init.constant_(self.final_linear.bias, 0.0)

    def forward(self, user, item, label):
        gmf_user_embed = self.gmf_user_embedding(user)
        gmf_item_embed = self.gmf_item_embedding(item)

        mlp_user_embed = self.mlp_user_embedding(user)
        mlp_item_embed = self.mlp_item_embedding(item)
                
        gmf_concat_embed = gmf_user_embed * gmf_item_embed
        
        mlp_concat_emebd = torch.cat([mlp_user_embed, mlp_item_embed], dim=1)
        for idx in range(len(conf.mlp_dim_list)-1):
            mlp_concat_emebd = torch.relu(self.linears[idx](mlp_concat_emebd))
        
        
        final_embed = torch.cat([gmf_concat_embed, mlp_concat_emebd], dim=1)
        prediction = self.final_linear(final_embed).view(-1) + conf.avg_rating

        #prediction = torch.sum(output_emb, 1, keepdims=True)

        obj_loss = self.obj_function(prediction, label) 
        mse_loss = self.loss_function(prediction, label)
        
        return prediction, obj_loss, mse_loss