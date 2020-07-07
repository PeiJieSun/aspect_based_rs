vocab_sz = 16602
word_dim = 200
batch_size = 50

learning_rate = 1e-3

train_epochs = 20

asp_dim = 15
num_neg_sent = 20

lr_lambda = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_pet'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path

seq_len = 11
sent_num = 8