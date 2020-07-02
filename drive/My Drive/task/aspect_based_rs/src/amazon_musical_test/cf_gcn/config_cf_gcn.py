num_users = 1429
num_items = 900
mf_dim = 32
word_dim = 512
hidden_dim = 512

learning_rate = 0.002
weight_decay = 0.02
vocab_sz = 12794
batch_size = 256
train_epochs = 30

num_words = 4276
encoder_word_dim = 200

rev_len = 31
seq_len = 11
user_seq_num = 8
item_seq_num = 13

avg_rating = 4.4848

root_path = '/content/drive/My Drive/task/aspect_based_rs/'
data_name = 'amazon_musical_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path

dropout = 0.1