data_dir = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_sports'
data_name = 'amazon_sports'
train_data_path = '%s/%s.train.data' % (data_dir, data_name)
val_data_path = '%s/%s.val.data' % (data_dir, data_name)
test_data_path = '%s/%s.test.data' % (data_dir, data_name)

surprise_train_data = open('/content/drive/My Drive/task/aspect_based_rs/data/%s/surprise.%s.rating.train' % (data_name, data_name), 'w')
surprise_val_data = open('/content/drive/My Drive/task/aspect_based_rs/data/%s/surprise.%s.rating.val' % (data_name, data_name), 'w')
surprise_test_data = open('/content/drive/My Drive/task/aspect_based_rs/data/%s/surprise.%s.rating.test' % (data_name, data_name), 'w')

f = open(train_data_path)
for line in f:
    line = eval(line)
    idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
    surprise_train_data.write('%d\t%d\t%d\n' % (user, item, rating))

f = open(val_data_path)
for line in f:
    line = eval(line)
    idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
    surprise_val_data.write('%d\t%d\t%d\n' % (user, item, rating))

f = open(test_data_path)
for line in f:
    line = eval(line)
    idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
    surprise_test_data.write('%d\t%d\t%d\n' % (user, item, rating))