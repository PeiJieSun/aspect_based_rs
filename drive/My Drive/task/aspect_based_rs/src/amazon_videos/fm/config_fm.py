num_users = 24302
num_items = 10672

avg_rating = 4.0883

embedding_dim = 32
lr_abae = 1.0
lr_rating = 1.0

lr = 2e-3

num_workers = 0

batch_size = 128
train_epochs = 20
weight_decay = 1e-3

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_clothing'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_clothing/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
model_path = '%s/out/model' % root_path