num_users = 4905
num_items = 2420

avg_rating = 4.3481

vocab_sz = 11683 + 3

u_max_r = 13
i_max_r = 32

word_dimension = 200

batch_size = 256
train_epochs = 30
weight_decay = 1e-3

embedding_dim = 32
lr_abae = 1.0
lr_rating = 1.0
learning_rate = 2e-3
drop_out = 0.5
num_negative_reviews = 1

kernel_size = 3
filters_num = 100

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_musical'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path