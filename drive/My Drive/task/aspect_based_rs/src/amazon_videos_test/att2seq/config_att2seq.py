word_dim = 512
hidden_dim = 512
mf_dim = 32

dropout = 0.1
learning_rate = 2e-3
batch_size = 256
train_epochs = 50

num_users = 24301
num_items = 10672
vocab_sz = 29736
rev_len = 31

root_path = '/content/drive/My Drive/task/aspect_based_rs/'
data_name = 'amazon_videos_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path