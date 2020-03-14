import numpy as np

import DataModule_mean as data_utils

# calculate the mean rating from all the train ratings
train_rating, val_rating, test_rating = data_utils.load_all()
mean_rating = np.mean(train_rating)

# train rating mean and calculate the rmse loss
train_mse_loss = np.sqrt(np.mean((train_rating - mean_rating)**2))

# val rating mean and calculate the rmse loss
val_mse_loss = np.sqrt(np.mean((val_rating - mean_rating)**2))

# test rating mean and calculate the rmse loss
test_mse_loss = np.sqrt(np.mean((test_rating - mean_rating)**2))

print('mean rating:%.4f' % mean_rating)
print('train mse loss:%.4f' % train_mse_loss)
print('val mse loss:%.4f' % val_mse_loss)
print('test mse loss:%.4f' % test_mse_loss)