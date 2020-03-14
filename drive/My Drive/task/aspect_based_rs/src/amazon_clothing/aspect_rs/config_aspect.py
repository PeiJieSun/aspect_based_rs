word_dimension = 50
aspect_dimension = 20

num_users = 39380
num_items = 23034
embedding_dim = 50

avg_rating = 4.2442

lr_abae = 0.01
lr_rating = 1.0

learning_rate = 0.001

vocab_sz = 15350
batch_size = 4096
train_epochs = 500
weight_decay = 0.001

num_negative_reviews = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_clothing'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_clothing/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
model_path = '%s/out/model' % root_path