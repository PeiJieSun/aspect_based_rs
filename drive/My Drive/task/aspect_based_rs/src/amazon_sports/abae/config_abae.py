vocab_sz = 22966
word_dimension = 50
hidden_size = 50
batch_size = 512

learning_rate = 1e-3

train_epochs = 20

aspect_dimension = 20
num_negative_reviews = 5

lr_lambda = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_sports'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
origin_file = '/content/drive/My Drive/datasets/amazon_sports/reviews_Sports_and_Outdoors_5.json.gz'
model_path = '%s/out/model' % root_path