word_dimension = 200
aspect_dimension = 32

common_dimension = word_dimension

num_users = 24302
num_items = 10672

avg_rating = 4.0883

embedding_dim = 32

vocab_sz = 35901 + 3

u_max_r = 9
i_max_r = 24


batch_size = 256
lr_abae = 1.0
lr_rating = 1.0

learning_rate = 2e-3
drop_out = 0.5
train_epochs = 30
weight_decay = 1e-3
num_negative_reviews = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_videos'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_reviews/reviews_Video_Games_5.json.gz'