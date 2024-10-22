weight_decay = 0.02
learning_rate = 0.002
mf_dim = 32

avg_rating = 4.0855

num_users = 24301
num_items = 10672

batch_size = 256

train_epochs = 20

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_videos_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path


#### parameters for probe: PMF
mf_dim = 32