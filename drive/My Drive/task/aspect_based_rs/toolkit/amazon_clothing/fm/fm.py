#!pip install git+https://github.com/coreylynch/pyFM

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm

# Read in data
def loadData(file_path):
    data = []
    y = []
    users=set()
    items=set()
    with open(file_path) as f:
        for line in f:
            line = eval(line)
            user, movieid, rating = line['user'], line['item'], line['rating']
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)
    return (data, np.array(y), users, items)

(train_data, y_train, train_users, train_items) = loadData("/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/amazon_clothing.train.data")
(val_data, y_val, val_users, val_items) = loadData("/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/amazon_clothing.val.data")
(test_data, y_test, test_users, test_items) = loadData("/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/amazon_clothing.test.data")
v = DictVectorizer()
X_train = v.fit_transform(train_data)

zX_train = v.transform(train_data)
X_val = v.transform(val_data)
X_test = v.transform(test_data)

# Build and train a Factorization Machine
fm = pylibfm.FM(num_factors=32, num_iter=30, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")

fm.fit(X_train, y_train)

# Evaluate Train
train_preds = fm.predict(zX_train)
print("Train RMSE: %.4f" % np.sqrt(np.mean((y_train-train_preds)**2)))

# Evaluate Val
val_preds = fm.predict(X_val)
print("Val RMSE: %.4f" % np.sqrt(np.mean((y_val-val_preds)**2)))

# Evaluate Test
test_preds = fm.predict(X_test)
print("Test RMSE: %.4f" % np.sqrt(np.mean((y_test-test_preds)**2)))

#import pdb; pdb.set_trace()




