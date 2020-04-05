import os

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split

import numpy as np

# path to dataset file
train_file_path = os.path.expanduser('/content/drive/My Drive/task/aspect_based_rs/data/amazon_tools/amazon_tools.train.rating')
val_file_path = os.path.expanduser('/content/drive/My Drive/task/aspect_based_rs/data/amazon_tools/amazon_tools.val.rating')
test_file_path = os.path.expanduser('/content/drive/My Drive/task/aspect_based_rs/data/amazon_tools/amazon_tools.test.rating')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating', sep='\t')

train_data = Dataset.load_from_file(train_file_path, reader=reader)
val_data = Dataset.load_from_file(val_file_path, reader=reader)
test_data = Dataset.load_from_file(test_file_path, reader=reader)

#trainset, testset = train_test_split(data, test_size=.10)

trainset, valset, testset = train_data.construct_trainset(train_data.raw_ratings), \
    val_data.construct_testset(val_data.raw_ratings), test_data.construct_testset(test_data.raw_ratings)

x_trainset = train_data.construct_testset(train_data.raw_ratings)

# We'll use the famous SVD algorithm.
algo = SVD(n_factors=100, biased=True, verbose=True)

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
train_predictions = algo.test(x_trainset)
val_predictions = algo.test(valset)
test_predictions = algo.test(testset)

# Then compute RMSE
print('Train', end = ''); accuracy.rmse(train_predictions)
print('Val', end = ''); accuracy.rmse(val_predictions)
print('Test', end = ''); accuracy.rmse(test_predictions)

train_ratings, val_ratings, test_ratings = [], [], []
for x in train_predictions:
    train_ratings.append(x.est)
for x in val_predictions:
    val_ratings.append(x.est)
for x in test_predictions:
    test_ratings.append(x.est)
print('train rating mean:%.4f, var:%.4f' % (np.mean(train_ratings), np.var(train_ratings)))
print('val rating mean:%.4f, var:%.4f' % (np.mean(val_ratings), np.var(val_ratings)))
print('test rating mean:%.4f, var:%.4f' % (np.mean(test_ratings), np.var(test_ratings)))