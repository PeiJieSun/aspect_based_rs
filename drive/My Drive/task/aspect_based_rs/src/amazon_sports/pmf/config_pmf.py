weight_decay = 0.001

learning_rate = 0.001

embedding_dim = 50

avg_rating = 4.3926

num_users = 35592
num_items = 18357

batch_size = 256

train_epochs = 1000

vocab_size = 22966

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_sports'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_sports/reviews_Sports_and_Outdoors_5.json.gz'