num_users = 39380
num_items = 23033
mf_dimension = 32
word_dimension = 200
hidden_dimension = 256
learning_rate = 0.002
weight_decay = 0.01
vocab_sz = 15353
batch_size = 256
train_epochs = 30
sequence_length = 29

avg_rating = 4.2442

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_clothing'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_clothing/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
model_path = '%s/out/model' % root_path