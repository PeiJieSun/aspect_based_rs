import config as conf

import re
import json
import gzip
import nltk
nltk.download('punkt')

from gensim.models import Word2Vec

import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

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

##### 
# Read file, and stroe the lines with file_data(dict)
#####
file_data = {}
total_words = list()
g = gzip.open(conf.origin_file, 'r')
review_text = []
for line_idx, line in enumerate(tqdm(g)):
    line = eval(line)
    sentences = nltk.tokenize.sent_tokenize(line['reviewText'].lower())
    summary_tokens = nltk.word_tokenize(line['summary'].lower())
    sent_token_list = []
    if len(sentences) > 0:
        for sent in sentences:
            sent_tokens = nltk.word_tokenize(sent)
            if len(sent_tokens) > 0:
                sent_token_list.append(sent_tokens)
                total_words.extend(sent_tokens)
    str_review = clean_str(line['reviewText'].encode('ascii', 'ignore').decode('ascii'))
    review_text.append(str_review)
    file_data[line_idx] = [line['reviewerID'], line['asin'], float(line['overall']), summary_tokens, sent_token_list]

##### 
# Construct vocabulary for review generation with size=30000;
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

g_vocab, g_vocab_decoder = {}, {}
for word_id, word in enumerate(word_list):
    g_vocab[word] = word_id + 3
    g_vocab_decoder[word_id + 3] = word
g_vocab['padx'] = 0; g_vocab_decoder[0] = 'padx'
g_vocab['sos'] = 1; g_vocab_decoder[1] = 'sos'
g_vocab['eos'] = 2; g_vocab_decoder[2] = 'eos'

np.save('%s/%s.g_vocab' % (conf.target_path, conf.data_name), g_vocab)
np.save('%s/%s.g_vocab_decoder' % (conf.target_path, conf.data_name), g_vocab_decoder)

##### 
# Construct vocabulary for abae review and rating prediction with size=30000;
# 1. Remove stop words;
# 2. Use word2vec words;
#####
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(review_text)
abae_vocab_without_stop_words = vectorizer.vocabulary_

abae_vocab, abae_vocab_decoder = {}, {}
w2v_model = Word2Vec.load("%s/%s.wv.model" % (conf.target_path, conf.data_name))
for word_id in range(len(w2v_model.wv.vocab)):
    word = w2v_model.wv.index2entity[word_id]
    abae_vocab[word] = word_id + 1
    abae_vocab_decoder[word_id + 1] = word
abae_vocab['pad'] = 0; abae_vocab_decoder[0] = 'pad'

np.save('%s/%s.abae_vocab' % (conf.target_path, conf.data_name), abae_vocab)
np.save('%s/%s.abae_vocab_decoder' % (conf.target_path, conf.data_name), abae_vocab_decoder)

#####
# Split data with their line idx
#####
line_idx_list = []
line_idx_list = list(file_data.keys())
np.random.shuffle(line_idx_list)
train_idx_list, val_idx_list, test_idx_list = \
    np.split(line_idx_list, [int(0.8*len(line_idx_list)), int(0.9*len(line_idx_list))])

train_data = open('%s/%s.train.data' % (conf.target_path, conf.data_name), 'w')
val_data = open('%s/%s.val.data' % (conf.target_path, conf.data_name), 'w')
test_data = open('%s/%s.test.data' % (conf.target_path, conf.data_name), 'w')

sent_len_list, sent_num_list, summary_len_list = [], [], []

#####
# Convert User & Product String ID into integer IDs
#####
print('Start to write lines to train data...')
user_idx_dict, item_idx_dict = {}, {}
for line_idx in tqdm(train_idx_list):
    [o_user_id, o_item_id, rating, summary_tokens, sent_token_list] = file_data[line_idx]
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
    
    abae_word_id_list = []
    for sent_tokens in sent_token_list:
        word_id_list = []
        for word in sent_tokens:
            if word in abae_vocab_without_stop_words and word in abae_vocab:
                word_id_list.append(abae_vocab[word])
        abae_word_id_list.append(word_id_list)

        sent_len_list.append(len(word_id_list))
    sent_num_list.append(len(abae_word_id_list))

    sent_word_id_list = []
    for sent_tokens in sent_token_list:
        word_id_list = []
        for word in sent_tokens:
            if word in g_vocab:
                sent_word_id_list.append(g_vocab[word])

    summary_word_id_list = []
    for word in summary_tokens:
        if word in g_vocab:
            summary_word_id_list.append(g_vocab[word])

    summary_len_list.append(len(summary_word_id_list))

    train_data.write('%s\n' % json.dumps({'user': n_user_id, \
        'item': n_item_id, 'rating': rating, 'abae_review': abae_word_id_list,\
        'g_review': sent_word_id_list, 'summary':summary_word_id_list}))

print('Start to write lines to val data...')
for line_idx in tqdm(val_idx_list):
    [o_user_id, o_item_id, rating, summary_tokens, sent_token_list] = file_data[line_idx]
    if o_user_id in user_idx_dict and o_item_id in item_idx_dict:
        n_user_id = user_idx_dict[o_user_id]
        n_item_id = item_idx_dict[o_item_id]
    
    abae_word_id_list = []
    for sent_tokens in sent_token_list:
        word_id_list = []
        for word in sent_tokens:
            if word in abae_vocab_without_stop_words and word in abae_vocab:
                word_id_list.append(abae_vocab[word])
        abae_word_id_list.append(word_id_list)

    sent_word_id_list = []
    for sent_tokens in sent_token_list:
        word_id_list = []
        for word in sent_tokens:
            if word in g_vocab:
                sent_word_id_list.append(g_vocab[word])

    summary_word_id_list = []
    for word in summary_tokens:
        if word in g_vocab:
            summary_word_id_list.append(g_vocab[word])
    val_data.write('%s\n' % json.dumps({'user': n_user_id, \
        'item': n_item_id, 'rating': rating, 'abae_review': abae_word_id_list,\
        'g_review': sent_word_id_list, 'summary':summary_word_id_list}))

print('Start to write lines to test data...')
for line_idx in tqdm(test_idx_list):
    [o_user_id, o_item_id, rating, summary_tokens, sent_token_list] = file_data[line_idx]
    if o_user_id in user_idx_dict and o_item_id in item_idx_dict:
        n_user_id = user_idx_dict[o_user_id]
        n_item_id = item_idx_dict[o_item_id]
    
    abae_word_id_list = []
    for sent_tokens in sent_token_list:
        word_id_list = []
        for word in sent_tokens:
            if word in abae_vocab_without_stop_words and word in abae_vocab:
                word_id_list.append(abae_vocab[word])
        abae_word_id_list.append(word_id_list)

    sent_word_id_list = []
    for sent_tokens in sent_token_list:
        word_id_list = []
        for word in sent_tokens:
            if word in g_vocab:
                sent_word_id_list.append(g_vocab[word])

    summary_word_id_list = []
    for word in summary_tokens:
        if word in g_vocab:
            summary_word_id_list.append(g_vocab[word])
    test_data.write('%s\n' % json.dumps({'user': n_user_id, \
        'item': n_item_id, 'rating': rating, 'abae_review': abae_word_id_list,\
        'g_review': sent_word_id_list, 'summary':summary_word_id_list}))

sent_len_list.sort()
sent_num_list.sort()
summary_len_list.sort()

print('sentence max length:%d' % sent_len_list[int(0.85*len(sent_len_list))])
print('review max sentence number:%d' % sent_num_list[int(0.85*len(sent_num_list))])
print('max summary length:%d' % summary_len_list[int(0.85*len(summary_len_list))])