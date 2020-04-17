num_users = 1428
num_items = 900

avg_rating = 4.4752

vocab_sz = 4275 + 3

u_max_r = 7
i_max_r = 11

max_review_len = 67

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
data_name = 'amazon_musical'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path