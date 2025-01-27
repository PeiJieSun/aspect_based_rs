num_users = 5541
num_items = 3568

avg_rating = 4.2230

vocab_sz = 23470 + 3

u_max_r = 11
i_max_r = 22

max_review_len = 161

word_dimension = 200
aspect_dimension = 20
embedding_dim = 32

batch_size = 128
lr_abae = 1.0
lr_rating = 1.0

learning_rate = 2e-3
drop_out = 0.5
train_epochs = 30
weight_decay = 1e-3
num_negative_reviews = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_digital'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path