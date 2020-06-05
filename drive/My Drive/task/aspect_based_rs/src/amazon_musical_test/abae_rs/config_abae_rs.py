vocab_sz = 4276
word_dim = 200
batch_size = 256

learning_rate = 1e-3

train_epochs = 60

asp_dim = 15
num_neg_sent = 4

lr_lambda = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_musical_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path

seq_len = 11
sent_num = 8
summary_len = 8

user_seq_num = 8
item_seq_num = 13

num_users = 1429
num_items = 900

mf_dim = 32

dropout = 0.5
avg_rating = 4.4848