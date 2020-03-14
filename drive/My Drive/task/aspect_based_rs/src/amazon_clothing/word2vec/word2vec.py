import json
import gzip

from gensim import utils

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file_path = '/content/drive/My Drive/datasets/amazon_clothing/'
origin_file = '%s/reviews_Clothing_Shoes_and_Jewelry_5.json.gz' % file_path

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
      g = gzip.open(origin_file, 'r')
      for line in g:
        yield utils.simple_preprocess(eval(line)['reviewText'])
            
import gensim.models

sentences = MyCorpus()
print('Start to train word embedding')
model = gensim.models.Word2Vec(sentences, min_count=10, size=50, workers=12, iter=10)
model.save('/content/drive/My Drive/task/aspect_based_rs/data/amazon_clothing/amazon_clothing.wv.model')