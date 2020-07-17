vocab_sz = 35902
word_dim = 200
batch_size = 256

learning_rate = 0.002

train_epochs = 30

asp_dim = 15
num_neg_sent = 1

lr_lambda = 1

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_videos_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path

seq_len = 14
sent_num = 17
summary_len = 8

user_seq_num = 10
item_seq_num = 27

num_users = 24301
num_items = 10672

mf_dim = 32

dropout = 0.5
avg_rating = 4.0855


#### NCF parameters ####
gmf_embed_dim = 32
mlp_embed_dim = 32

mlp_dim_list = [64, 128, 64, 32]

weight_decay = 0.02