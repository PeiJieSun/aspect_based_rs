import numpy as np 
from collections import defaultdict

from copy import deepcopy

PAD = 0; SOS = 1; EOS = 2

data_name = 'amazon_sports'
target_path = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_sports'
train_data_path = '%s/%s.train.data' % (target_path, data_name)
val_data_path = '%s/%s.val.data' % (target_path, data_name)
test_data_path = '%s/%s.test.data' % (target_path, data_name)

def load_all():
    train_rating = []
    f = open(train_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
        train_rating.append(rating)

    val_rating = []
    f = open(val_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
        val_rating.append(rating)
    
    test_rating = []
    f = open(test_data_path)
    for line in f:
        line = eval(line)
        idx, user, item, rating = line['idx'], line['user'], line['item'], line['rating']
        test_rating.append(rating)

    return train_rating, val_rating, test_rating
