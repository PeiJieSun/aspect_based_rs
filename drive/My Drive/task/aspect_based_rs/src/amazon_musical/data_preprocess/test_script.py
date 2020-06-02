origin_file = '/content/drive/My Drive/datasets/amazon_reviews/reviews_Musical_Instruments_5.json.gz'
target_path = '/content/drive/My Drive/task/aspect_based_rs/data/amazon_musical_test'
data_name = 'amazon_musical_test'

import json
import gzip
import nltk
nltk.download('punkt')

import numpy as np
from tqdm import tqdm
from collections import Counter

##### 
#  Read file, and stroe the lines with file_data(dict)
#####
file_data = {}
total_words = list()
g = gzip.open(origin_file, 'r')
for line_idx, line in enumerate(tqdm(g)):
    line = eval(line)
    tokens = nltk.word_tokenize(line['reviewText'].lower())
    total_words.extend(tokens)
    if len(tokens) <= 100:
        file_data[line_idx] = [line['reviewerID'], line['asin'], float(line['overall']), tokens]

##### 
#  Construct Vocabulary with size=30000;
#####
# https://stackoverflow.com/questions/55464012/any-efficient-way-to-create-vocabulary-of-top-frequent-words-from-list-of-senten
c = dict(Counter(total_words))
words_frequency = list(c.values())
words_frequency.sort()
target_frequencey = words_frequency[-30000]
word_list = []
for word in c:
    if c[word] > target_frequencey:
        word_list.append(word)
vocab, vocab_decoder = {}, {}
for word_id, word in enumerate(word_list):
    vocab[word] = word_id + 3
    vocab_decoder[word_id + 3] = word
vocab['pad'] = 0; vocab_decoder[0] = 'pad'
vocab['sos'] = 1; vocab_decoder[1] = 'sos'
vocab['eos'] = 2; vocab_decoder[2] = 'eos'

np.save('%s/%s.vocab' % (target_path, data_name), vocab)
np.save('%s/%s.vocab_decoder' % (target_path, data_name), vocab_decoder)

#####
# Split data with their line idx
#####
line_idx_list = []
line_idx_list = list(file_data.keys())
np.random.shuffle(line_idx_list)
train_idx_list, val_idx_list, test_idx_list = np.split(line_idx_list, [int(0.8*len(line_idx_list)), int(0.9*len(line_idx_list))])

train_data = open('%s/%s.train.data' % (target_path, data_name), 'w')
val_data = open('%s/%s.val.data' % (target_path, data_name), 'w')
test_data = open('%s/%s.test.data' % (target_path, data_name), 'w')

#####
# Convert User & Product String ID into integer IDs
#####
print('Start to write lines to train data...')
user_idx_dict, item_idx_dict = {}, {}
for line_idx in tqdm(train_idx_list):
    o_user_id, o_item_id, rating, tokens = file_data[line_idx][0], \
        file_data[line_idx][1], file_data[line_idx][2], file_data[line_idx][3]
    if o_user_id in user_idx_dict:
        n_user_id = user_idx_dict[o_user_id]
    else:
        n_user_id = len(user_idx_dict.keys())
        user_idx_dict[o_user_id] = n_user_id
    if o_item_id in item_idx_dict:
        n_item_id = item_idx_dict[o_item_id]
    else:
        n_item_id = len(item_idx_dict.keys())
        item_idx_dict[o_item_id] = n_item_id
    word_id_list = []
    for word in tokens:
        if word in vocab:
            word_id_list.append(vocab[word])
    train_data.write('%s\n' % json.dumps({'user': n_user_id, \
        'item': n_item_id, 'rating': rating, 'g_review': word_id_list}))

print('Start to write lines to val data...')
for line_idx in tqdm(val_idx_list):
    o_user_id, o_item_id, rating, tokens = file_data[line_idx][0], \
        file_data[line_idx][1], file_data[line_idx][2], file_data[line_idx][3]
    if o_user_id in user_idx_dict and o_item_id in item_idx_dict:
        n_user_id = user_idx_dict[o_user_id]
        n_item_id = item_idx_dict[o_item_id]
    word_id_list = []
    for word in tokens:
        if word in vocab:
            word_id_list.append(vocab[word])
    val_data.write('%s\n' % json.dumps({'user': n_user_id, \
        'item': n_item_id, 'rating': rating, 'g_review': word_id_list}))

print('Start to write lines to test data...')
for line_idx in tqdm(test_idx_list):
    o_user_id, o_item_id, rating, tokens = file_data[line_idx][0], \
        file_data[line_idx][1], file_data[line_idx][2], file_data[line_idx][3]
    if o_user_id in user_idx_dict and o_item_id in item_idx_dict:
        n_user_id = user_idx_dict[o_user_id]
        n_item_id = item_idx_dict[o_item_id]
    word_id_list = []
    for word in tokens:
        if word in vocab:
            word_id_list.append(vocab[word])
    test_data.write('%s\n' % json.dumps({'user': n_user_id, \
        'item': n_item_id, 'rating': rating, 'g_review': word_id_list}))