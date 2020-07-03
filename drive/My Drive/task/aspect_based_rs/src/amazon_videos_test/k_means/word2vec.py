import json
import gzip

from gensim import utils

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import config as conf

import re

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

def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
      g = gzip.open(conf.origin_file, 'r')
      for line in g:
        #yield utils.simple_preprocess(clean_str(eval(line)['reviewText']))
        line = eval(line)
        yield utils.simple_preprocess(clean_str(line['reviewText']))
            
import gensim.models

sentences = MyCorpus()
print('Start to train word embedding')
model = gensim.models.Word2Vec(sentences, min_count=10, size=200, workers=12, iter=10)

check_dir('%s/%s.wv.model' % (conf.target_path, conf.data_name))
model.save('%s/%s.wv.model' % (conf.target_path, conf.data_name))