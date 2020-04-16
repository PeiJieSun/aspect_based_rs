import torch
import torch.nn as nn
import torch.nn.functional as F 

import config_deepconn as conf

class deepconn(nn.Module):
    def __init__(self):
        super(deepconn, self).__init__()

        # parameters for DeepCoNN
        self.user_word_embs = nn.Embedding(conf.vocab_sz, conf.word_dimension)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(conf.vocab_sz, conf.word_dimension)  # vocab_size * 300

        self.user_cnn = nn.Conv2d(1, conf.filters_num, (conf.kernel_size, conf.word_dimension))
        self.item_cnn = nn.Conv2d(1, conf.filters_num, (conf.kernel_size, conf.word_dimension))

        self.user_fc_linear = nn.Linear(conf.filters_num, conf.embedding_dim)
        self.item_fc_linear = nn.Linear(conf.filters_num, conf.embedding_dim)
        self.dropout = nn.Dropout(conf.drop_out)

        self.free_user_embedding = nn.Embedding(conf.num_users, conf.embedding_dim)
        self.free_item_embedding = nn.Embedding(conf.num_items, conf.embedding_dim)

        # parameters for FM
        self.user_embedding = nn.Embedding(conf.num_users, conf.embedding_dim)  # user/item num * 32
        self.item_embedding = nn.Embedding(conf.num_items, conf.embedding_dim)
        self.user_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False

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

    def reset_para(self):
        nn.init.uniform_(self.free_user_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.free_item_embedding.weight, a=-0.1, b=0.1)

        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        if False:
            w2v = torch.from_numpy(np.load(conf.w2v_path))
            self.user_word_embs.weight.data.copy_(w2v.cuda())
            self.item_word_embs.weight.data.copy_(w2v.cuda())
        else:
            nn.init.xavier_normal_(self.user_word_embs.weight)
            nn.init.xavier_normal_(self.item_word_embs.weight)

        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def forward(self, user, item, label, user_doc, item_doc):
        # Two CNN modules for user & item review-embedding representation        
        user_doc = self.user_word_embs(user_doc)
        item_doc = self.item_word_embs(item_doc)

        u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)

        u_fea = self.user_fc_linear(u_fea)
        i_fea = self.item_fc_linear(i_fea)
        u_out = self.dropout(u_fea) + self.free_user_embedding(user)
        i_out = self.dropout(i_fea) + self.free_item_embedding(item)

        #import pdb; pdb.set_trace()

        input_vec = torch.cat([u_out, i_out], 1)

        input_vec = self.dropout(input_vec)
        #import pdb; pdb.set_trace()

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
        obj_loss = mse_loss
        
        return obj_loss, rating_loss, prediction