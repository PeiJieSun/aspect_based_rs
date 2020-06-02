word_dimension = 200

drop_out = 0.5
weight_decay = 0.02

vocab_sz = 15350 + 3
batch_size = 256

learning_rate = 2e-3

train_epochs = 30

sequence_length = 31

num_users = 39380
num_items = 23034

mf_dimension = 100

hidden_dimension = 256

aspect_dimension = 20

hidden_size = 200
m = 64; n = 200; k = 20

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_clothing'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_clothing/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
model_path = '%s/out/model' % root_path