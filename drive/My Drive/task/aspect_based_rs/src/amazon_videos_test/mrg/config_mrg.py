word_dim = 512
hidden_dim = 512
mf_dim = 32
weight_decay = 0.02

dropout = 0.1
learning_rate = 0.002
batch_size = 256
train_epochs = 30

num_users = 24301
num_items = 10672
vocab_sz = 29736
rev_len = 31

avg_rating = 4.0855
mlp_dim_list = [64, 128, 64, 32]

root_path = '/content/drive/My Drive/task/aspect_based_rs/'
data_name = 'amazon_videos_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path

mlp_embed_dim = 32