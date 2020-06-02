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