word_dimension = 200
aspect_dimension = 32

num_users = 16637
num_items = 10217

avg_rating = 4.3657

embedding_dim = 32

vocab_sz = 16933 + 3

u_max_r = 8
i_max_r = 14


batch_size = 256
lr_abae = 1.0
lr_rating = 1.0

learning_rate = 2e-3
drop_out = 0.5
train_epochs = 30
weight_decay = 1e-3
num_negative_reviews = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_tools'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path