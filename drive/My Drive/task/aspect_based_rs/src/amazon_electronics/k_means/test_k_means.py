import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from time import time
from copy import deepcopy
from gensim.models import Word2Vec

import DataModule_aspect as data_utils
import config_aspect as conf

from Logging import Logging

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == '__main__':
    # prepare the data
    electronics_wv_model = Word2Vec.load("/content/drive/My Drive/task/aspect_based_rs/data/amazon_electronics/electronics.wv.model")

    x = torch.randn(len(electronics_wv_model.wv.vocab.keys()), electronics_wv_model.wv.vector_size)
    for word_id in range(len(electronics_wv_model.wv.vocab.keys())):
        #import pdb; pdb.set_trace()
        x[word_id] = torch.FloatTensor(electronics_wv_model.wv[electronics_wv_model.wv.index2entity[word_id]])

    x = x.view(x.shape[0], 1, -1)
    aspect_matrix = torch.FloatTensor().view(1, -1, x.shape[1])

    D_ij = ((x_i * c_j) / ((F.normalize(x_i, p=2, dim=2)) * (F.normalize(c_j, p=2, dim=2))).sum(-1)

    values, indices = torch.topk(D_ij)
