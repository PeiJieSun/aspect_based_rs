num_users = 1347
num_items = 875
mf_dimension = 32
word_dimension = 200
hidden_dimension = 256
learning_rate = 0.002
weight_decay = 0.01
vocab_sz = 15353
batch_size = 128
train_epochs = 50
sequence_length = 30

avg_rating = 4.2442

root_path = '/content/drive/My Drive/task/aspect_based_rs/'
data_name = 'amazon_musical_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path