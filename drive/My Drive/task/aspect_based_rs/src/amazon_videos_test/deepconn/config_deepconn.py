
word_dim = 200
num_users = 24301
num_items = 10672
embedding_dim = 32
vocab_sz = 35902

avg_rating = 4.0855

dropout = 0.5

batch_size = 256
train_epochs = 30
weight_decay = 0.02
learning_rate = 0.002

seq_len = 14
sent_num = 17
user_seq_num = 10
item_seq_num = 27

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_videos_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path

kernel_size = 3
filters_num = 100

mf_dim = 32
'''

num_users = 1429
num_items = 900

batch_size = 256
weight_decay = 0.02
learning_rate = 0.002
embedding_dim = 32

avg_rating = 4.4848

train_epochs = 20

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_musical_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path

mf_dim=32
'''