import os

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split

# path to dataset file
train_file_path = os.path.expanduser('/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/surprise.amazon_clothing.rating.train')
val_file_path = os.path.expanduser('/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/surprise.amazon_clothing.rating.val')
test_file_path = os.path.expanduser('/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/surprise.amazon_clothing.rating.test')

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

# We'll use the famous SVD algorithm.
algo = SVD(n_epochs=1, biased=True, verbose=True)

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
val_predictions = algo.test(valset)
test_predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(val_predictions)
accuracy.rmse(test_predictions)
import pdb; pdb.set_trace()