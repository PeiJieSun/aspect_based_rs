data_dir = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing'
data_name = 'amazon_clothing'
train_data_path = '%s/%s.train.data' % (data_dir, data_name)
val_data_path = '%s/%s.val.data' % (data_dir, data_name)
test_data_path = '%s/%s.test.data' % (data_dir, data_name)

surprise_train_data = open('/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/surprise.amazon_clothing.rating.train', 'w')
surprise_val_data = open('/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/surprise.amazon_clothing.rating.val', 'w')
surprise_test_data = open('/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/surprise.amazon_clothing.rating.test', 'w')

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