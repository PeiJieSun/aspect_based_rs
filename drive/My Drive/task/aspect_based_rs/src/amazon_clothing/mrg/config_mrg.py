word_dimension = 200

drop_out = 0.5
weight_decay = 1e-3

vocab_sz = 15350 + 3
hidden_size = 200
batch_size = 256

num_users = 39380
num_items = 23034

learning_rate = 1e-3

train_epochs = 30

sequence_length = 31

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_clothing'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_clothing/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
model_path = '%s/out/model' % root_path

embedding_dim = 200