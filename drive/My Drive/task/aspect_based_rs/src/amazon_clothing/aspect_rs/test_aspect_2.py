#  In order to veryify the mean and variance between user embed, item embed and review embed
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()

data = 

user_embed = 
item_embed = 
review_embed = 

user_embed_loss = []
item_embed_loss = []
for (user, item, embed) in review_embed:
    user_embed_loss.append(mse_loss(embed, user_embed[user]))
    item_embed_loss.append(mse_loss(embed, item_embed[item]))

