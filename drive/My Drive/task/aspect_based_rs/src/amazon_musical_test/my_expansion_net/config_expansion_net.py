'''
word_dim = 512
att_dim = 64
aspect_dim = 15
dropout = 0.1
vocab_sz = 12793
batch_size = 256
learning_rate = 2e-3
hidden_dim = 512

train_epochs = 100

rev_len = 31

num_users = 1429
num_items = 900


root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_musical_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_reviews/reviews_Musical_Instruments_5.json.gz'
model_path = '%s/out/model' % root_path

seq_len = 11
seq_num = 8
sum_len = 8
'''



word_dim = 512
hidden_dim = 512

att_dim = 64
aspect_dim = 15

dropout = 0.1
batch_size = 256
train_epochs = 50
learning_rate = 0.002

abae_vocab_sz = 4276
gen_vocab_sz = 12794

vocab_sz = 12794

num_users = 1429
num_items = 900
seq_len = 11
seq_num = 8
sum_len = 8
rev_len = 31

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_musical_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_reviews/reviews_Musical_Instruments_5.json.gz'
model_path = '%s/out/model' % root_path


mf_dim = 32
avg_rating = 4.4848
weight_decay = 0.02