import gzip
import json

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from gensim.models import Word2Vec
from gensim import utils

import config as conf 

PAD = 0; SOS = 1; EOS = 2

def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()

check_dir('%s/%s.train.data' % (conf.target_path, conf.data_name))
train_data = open('%s/%s.train.data' % (conf.target_path, conf.data_name), 'w')
val_data = open('%s/%s.val.data' % (conf.target_path, conf.data_name), 'w')
test_data = open('%s/%s.test.data' % (conf.target_path, conf.data_name), 'w')

train_review_embedding_path = '%s/%s.train.review_embedding' % (conf.target_path, conf.data_name)
val_review_embedding_path = '%s/%s.val.review_embedding' % (conf.target_path, conf.data_name)
test_review_embedding_path = '%s/%s.test.review_embedding' % (conf.target_path, conf.data_name)
check_dir(train_review_embedding_path)

# first is generate the train / val / test data, then write them to the files.
train_data_list, val_data_list, test_data_list = [], [], []
train_user, train_item = (), ()
user_idx_dict, item_idx_dict = {}, {}

review_text = []
g = gzip.open(conf.origin_file, 'r')
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
        str_review = clean_str(line['reviewText'].encode('ascii', 'ignore').decode('ascii'))
        review_text.append(str_review)
print('data read complete.')

P_REVIEW = 0.85

vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(review_text)
vocab_without_stop_words = vectorizer.vocabulary_

review_len = []
for value in train_data_list:
    review = utils.simple_preprocess(value[3])
    tmp_review = []
    for word in review:
        if word in vocab_without_stop_words:
           tmp_review.append(word)
    review_len.append(len(tmp_review))

x = np.sort(review_len)
xLen = len(x)
MAX_SENTENCE_LENGTH = x[int(P_REVIEW * xLen) - 1]

print('max review length:%d' % MAX_SENTENCE_LENGTH)


w2v_model = Word2Vec.load("%s/%s.wv.model" % (conf.target_path, conf.data_name))
vocab_w2v = set(w2v_model.wv.index2entity[:40000])
# write train data set to file
idx = 0
train_review_embedding = {}
for value in train_data_list:
    user, item, rating = user_idx_dict[value[0]], item_idx_dict[value[1]], value[2]
    review_text = utils.simple_preprocess(value[3])
    if len(review_text) > 0:
        review, review_embedding = [], []
        for word in review_text:
            if word in vocab_without_stop_words and word in vocab_w2v:
                review.append(w2v_model.wv.vocab[word].index + 3)
                review_embedding.append(w2v_model.wv[word])
        if len(review) > 0: 
            if len(review) > MAX_SENTENCE_LENGTH:
                x_review = review[:MAX_SENTENCE_LENGTH]
                review = review[:MAX_SENTENCE_LENGTH]
                review_embedding = np.mean(review_embedding[:MAX_SENTENCE_LENGTH], axis=0)
            else:
                x_review = review
                review.extend([PAD]*(MAX_SENTENCE_LENGTH-len(review)))
                review_embedding = np.mean(review_embedding, axis=0)
            train_data.write('%s\n' % json.dumps({'idx': idx, 'user': user, 'item': item, 'rating': rating, 'review': review, 'x_review': x_review}))
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
                if word in vocab_without_stop_words and word in vocab_w2v:
                    review.append(w2v_model.wv.vocab[word].index + 3)
                    review_embedding.append(w2v_model.wv[word])
            if len(review) > 0:
                if len(review) > MAX_SENTENCE_LENGTH:
                    review = review[:MAX_SENTENCE_LENGTH]
                    review_embedding = np.mean(review_embedding[:MAX_SENTENCE_LENGTH], axis=0)
                else:
                    review.extend([PAD]*(MAX_SENTENCE_LENGTH-len(review)))
                    review_embedding = np.mean(review_embedding, axis=0)
                val_data.write('%s\n' % json.dumps({'idx': idx, 'user': user, 'item': item, 'rating': rating, 'review': review, 'x_review': x_review}))
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
                if word in vocab_without_stop_words and word in vocab_w2v:
                    review.append(w2v_model.wv.vocab[word].index + 3)
                    review_embedding.append(w2v_model.wv[word])
            if len(review) > 0:
                if len(review) > MAX_SENTENCE_LENGTH:
                    review = review[:MAX_SENTENCE_LENGTH]
                    review_embedding = np.mean(review_embedding[:MAX_SENTENCE_LENGTH], axis=0)
                else:
                    review.extend([PAD]*(MAX_SENTENCE_LENGTH-len(review)))
                    review_embedding = np.mean(review_embedding, axis=0)
                test_data.write('%s\n' % json.dumps({'idx': idx,'user': user, 'item': item, 'rating': rating, 'review': review, 'x_review': x_review}))
                test_review_embedding[idx] = review_embedding
                idx += 1
np.save(test_review_embedding_path, test_review_embedding)