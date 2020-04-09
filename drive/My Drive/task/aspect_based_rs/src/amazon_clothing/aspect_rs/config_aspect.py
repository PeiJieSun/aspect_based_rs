word_dimension = 200
aspect_dimension = 64

common_dimension = word_dimension

num_users = 39380
num_items = 23034
embedding_dim = 32

avg_rating = 4.2442

lr_abae = 1.0
lr_rating = 1.0

learning_rate = 2e-3

drop_out = 0.5

vocab_sz = 15350 + 3
batch_size = 256
train_epochs = 30
weight_decay = 1e-3

u_max_r = 7
i_max_r = 14



num_negative_reviews = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_clothing'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_clothing/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
model_path = '%s/out/model' % root_path