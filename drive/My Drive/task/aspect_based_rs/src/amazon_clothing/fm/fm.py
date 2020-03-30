import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_fm as conf 

class fm(nn.Module):
    def __init__(self):
        super(fm, self).__init__()
        self.user_embedding = nn.Embedding(conf.num_users, conf.embedding_dim)  # user/item num * 32
        self.item_embedding = nn.Embedding(conf.num_items, conf.embedding_dim)
        
        dim = conf.embedding_dim * 2
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(conf.num_users, 1))
        self.b_items = nn.Parameter(torch.randn(conf.num_items, 1))

        self.mse_func_1 = nn.MSELoss(reduction='none')
        self.mse_func_2 = nn.MSELoss()

        self.reset_para()

    def reset_para(self):
        nn.init.uniform_(self.user_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.item_embedding.weight, a=-0.1, b=0.1)

        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def forward(self, uids, iids, labels):
        u_fea = self.user_embedding(uids)
        i_fea = self.item_embedding(iids)

        #import pdb; pdb.set_trace()
        u_out = u_fea.view(-1, 1, conf.embedding_dim)
        i_out = i_fea.view(-1, 1, conf.embedding_dim)

        u_out = u_out.reshape(u_out.size(0), -1)
        i_out = i_out.reshape(i_out.size(0), -1)

        input_vec = torch.cat([u_out, i_out], 1)

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2) + fm_linear_part + 4.2442 + self.b_users[uids] + self.b_items[iids]

        prediction = fm_output.squeeze(1)
        mse_loss = self.mse_func_1(prediction, labels)
        obj_loss = self.mse_func_2(prediction, labels)
        #import pdb; pdb.set_trace()
        return prediction, obj_loss, mse_loss 