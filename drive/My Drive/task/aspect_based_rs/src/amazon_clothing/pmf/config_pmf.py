weight_decay = 0.02

learning_rate = 0.005

embedding_dim = 100

avg_rating = 4.2442

num_users = 39380
num_items = 23033

batch_size = 1024

train_epochs = 20

vocab_size = 15350

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_clothing'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_clothing/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
model_path = '%s/out/model' % root_path