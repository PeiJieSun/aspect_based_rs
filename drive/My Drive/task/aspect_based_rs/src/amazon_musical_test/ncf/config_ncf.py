weight_decay = 0.02
learning_rate = 0.002

gmf_embed_dim = 32
mlp_embed_dim = 32

mlp_dim_list = [64, 128, 64, 32]

avg_rating = 4.4867

num_users = 1429
num_items = 900

batch_size = 256

train_epochs = 20

root_path = '/content/drive/My Drive/task/aspect_based_rs'
data_name = 'amazon_musical_test'
target_path = '%s/data/%s' % (root_path, data_name)
out_path = '%s/out/%s' % (root_path, data_name)
model_path = '%s/out/model' % root_path


#### parameters for probe: PMF
mf_dim = 32