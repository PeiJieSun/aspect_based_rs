import os, sys, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import defaultdict
from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_abae as data_utils
import config_abae as conf

from Logging import Logging

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == '__main__':
    aspect_params = torch.load('/content/drive/My Drive/task/aspect_based_rs/out/amazon_musical_test/train_amazon_musical_test_abae_id_X1.mod')
    c = aspect_params['transform_T.weight'].transpose(0, 1) # (aspect_dimesion, word_dimension)

    k_means_weight = np.load('%s/%s.k_means_15.npy' % (conf.target_path, conf.data_name))
    #c = torch.FloatTensor(k_means_weight).cuda() # (aspect_dimesion, word_dimension)

    x = aspect_params['word_embedding.weight'] # (num_words, word_dimension)

    x_i = F.normalize(x[:, None, :], p=2, dim=2) # (num_words, 1, word_dimension)
    c_j = F.normalize(c[None, :, :], p=2, dim=2) # (1, aspect_dimesion, word_dimension)
    
    D_ij = torch.transpose((x_i * c_j).sum(-1), 0, 1) # (aspect_dimesion, num_words)

    K = 10
    values, indices = torch.topk(D_ij, K) # (aspect_dimesion, K)

    word_embedding = Word2Vec.load('%s/%s.wv.model' % (conf.target_path, conf.data_name))

    print(indices)
    #import pdb; pdb.set_trace()

    word_dict = np.load('/content/drive/My Drive/task/aspect_based_rs/data/amazon_musical_test/amazon_musical_test.abae_vocab_decoder.npy', allow_pickle=True).item()


    for idx, word_idx_list in enumerate(indices):
        aspect_word_list = 'aspect_%d: ' % (idx+1)
        for word_idx in word_idx_list:
            #aspect_word_list += '%s, ' % word_embedding.wv.index2entity[word_idx.item()-1]
            aspect_word_list += '%s, ' % word_dict[word_idx.item()]
        print(aspect_word_list)

    import sys; sys.exit(0)