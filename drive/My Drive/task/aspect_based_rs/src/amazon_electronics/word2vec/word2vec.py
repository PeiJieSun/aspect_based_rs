import json
import gzip

from gensim import utils

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file_path = '/content/drive/My Drive/datasets/amazon_electronics/'
origin_file = '%s/reviews_Electronics_5.json.gz' % file_path

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
      g = gzip.open(origin_file, 'r')
      for line in g:
        yield utils.simple_preprocess(eval(line)['reviewText'])
            
import gensim.models

sentences = MyCorpus()
print('Start to train word embedding')
model = gensim.models.Word2Vec(sentences, min_count=3, size=50, workers=12)
model.save('/content/drive/My Drive/task/aspect_based_rs/amazon_electronics/electronics.wv.model')