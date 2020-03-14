import gzip
import json

import numpy as np

from gensim.models import Word2Vec
from gensim import utils

PAD = 0; SOS = 1; EOS = 2

w2v_model = Word2Vec.load("/content/drive/My Drive/task/aspect_based_rs/data/amazon_electronics/electronics.wv.model")
vocab = set(w2v_model.wv.index2entity[:40000])

data_name = 'amazon_electronics'
file_path = '/content/drive/My Drive/datasets/amazon_electronics/'
origin_file = '%s/reviews_Electronics_5.json.gz' % file_path

target_path = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_electronics'
train_data = open('%s/%s.train.data' % (target_path, data_name), 'w')
val_data = open('%s/%s.val.data' % (target_path, data_name), 'w')
test_data = open('%s/%s.test.data' % (target_path, data_name), 'w')

train_review_embedding_path = '%s/%s.train.review_embedding' % (target_path, data_name)
val_review_embedding_path = '%s/%s.val.review_embedding' % (target_path, data_name)
test_review_embedding_path = '%s/%s.test.review_embedding' % (target_path, data_name)

# first is generate the train / val / test data, then write them to the files.
train_data_list, val_data_list, test_data_list = [], [], []
train_user, train_item = (), ()
user_idx_dict, item_idx_dict = {}, {}

g = gzip.open(origin_file, 'r')
for idx, line in enumerate(g):
    line = eval(line)
    if idx % 8 == 0:
        val_data_list.append([line['reviewerID'], line['asin'], line['overall'], line['reviewText']])
    elif idx % 9 == 0:
        test_data_list.append([line['reviewerID'], line['asin'], line['overall'], line['reviewText']])
    else:
        if line['reviewerID'] not in user_idx_dict:
            user_idx_dict[line['reviewerID']] = len(user_idx_dict.keys())
        if line['asin'] not in item_idx_dict:
            item_idx_dict[line['asin']] = len(item_idx_dict.keys())
        train_data_list.append([line['reviewerID'], line['asin'], line['overall'], line['reviewText']])
print('data read complete.')


# write train data set to file
idx = 0
train_review_embedding = {}
for value in train_data_list:
    user, item, rating = user_idx_dict[value[0]], item_idx_dict[value[1]], value[2]
    review_text = utils.simple_preprocess(value[3])
    if len(review_text) > 0:
        review, review_embedding = [], []
        for word in review_text:
            if word in vocab:
                review.append(w2v_model.wv.vocab[word].index + 3)
                review_embedding.append(w2v_model.wv[word])
        if len(review) > 0: 
            if len(review) > 30:
                review = review[:30]
                review_embedding = np.mean(review_embedding[:30], axis=0)
            else:
                review.extend([PAD]*(30-len(review)))
                review_embedding = np.mean(review_embedding, axis=0)
            train_data.write('%s\n' % json.dumps({'idx': idx, 'user': user, 'item': item, 'rating': rating, 'review': review}))
            train_review_embedding[idx] = review_embedding
            idx += 1
np.save(train_review_embedding_path, train_review_embedding)

# write test data set to file
idx = 0
val_review_embedding = {}
for value in val_data_list:
    reviewerID, asin = value[0], value[1]
    if reviewerID in user_idx_dict and asin in item_idx_dict:
        user, item, rating = user_idx_dict[reviewerID], item_idx_dict[asin], value[2]
        review_text = utils.simple_preprocess(value[3])
        if len(review_text) > 0:
            review, review_embedding = [], []
            for word in review_text:
                if word in vocab:
                    review.append(w2v_model.wv.vocab[word].index + 3)
                    review_embedding.append(w2v_model.wv[word])
            if len(review) > 0:
                if len(review) > 30:
                    review = review[:30]
                    review_embedding = np.mean(review_embedding[:30], axis=0)
                else:
                    review.extend([PAD]*(30-len(review)))
                    review_embedding = np.mean(review_embedding, axis=0)
                val_data.write('%s\n' % json.dumps({'idx': idx, 'user': user, 'item': item, 'rating': rating, 'review': review}))
                val_review_embedding[idx] = review_embedding
                idx += 1
np.save(val_review_embedding_path, val_review_embedding)

# write test data set to file
idx = 0
test_review_embedding = {}
for value in test_data_list:
    reviewerID, asin = value[0], value[1]
    if reviewerID in user_idx_dict and asin in item_idx_dict:
        user, item, rating = user_idx_dict[reviewerID], item_idx_dict[asin], value[2]
        review_text = utils.simple_preprocess(value[3])
        if len(review_text) > 0:
            review, review_embedding = [], []
            for word in review_text:
                if word in vocab:
                    review.append(w2v_model.wv.vocab[word].index + 3)
                    review_embedding.append(w2v_model.wv[word])
            if len(review) > 0:
                if len(review) > 30:
                    review = review[:30]
                    review_embedding = np.mean(review_embedding[:30], axis=0)
                else:
                    review.extend([PAD]*(30-len(review)))
                    review_embedding = np.mean(review_embedding, axis=0)
                test_data.write('%s\n' % json.dumps({'idx': idx,'user': user, 'item': item, 'rating': rating, 'review': review}))
                test_review_embedding[idx] = review_embedding
                idx += 1
np.save(test_review_embedding_path, test_review_embedding)