import nltk
nltk.download('stopwords')

import time

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import numpy as np

from gensim.models import Word2Vec
from nltk.corpus import stopwords 

from collections import defaultdict

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niter):
        c_j = c[None, :, :]  # (1, Nclusters, D)

        # euclidean distance
        #D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances

        # cosine distance
        x_tmp = F.normalize(x_i, p=2, dim=2); c_tmp = F.normalize(c_j, p=2, dim=2)
        # x_tmp * c_tmp: (Npoints, Nclusters, D)
        D_ij = (x_tmp * c_tmp).sum(-1) # (Npoints, Nclusters)

        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        
        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print('K-means example with {:,} points in dimension {:,}, K = {:,}:'.format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                Niter, end - start, Niter, (end-start) / Niter))

    # c: (Nclusters, D)
    return cl, c

if use_cuda:
    stop_words = set(stopwords.words('english'))
    word_dict = defaultdict()

    # prepare the data x
    electronics_wv_model = Word2Vec.load('/content/drive/My Drive/task/aspect_based_rs/data/amazon_movies_tv/amazon_movies_tv.wv.model')

    x = []; idx = 0
    for word_id in range(40000):
        word = electronics_wv_model.wv.index2entity[word_id]
        x.append(electronics_wv_model.wv[word])
        word_dict[idx] = word
        idx += 1
        '''
        if word not in stop_words:
            x.append(electronics_wv_model.wv[word])
            word_dict[idx] = word
            idx += 1
        '''
    x = torch.FloatTensor(x).cuda() # (Npoints, D)

    # get the cluster center vectors c 
    K = 20
    clx, c = KMeans(x, K)
    np.save('/content/drive/My Drive/task/aspect_based_rs/data/amazon_movies_tv/amazon_movies_tv.k_means', tensorToScalar(c))
    
    # output the related words to each aspect
    x_i = F.normalize(x[:, None, :], p=2, dim=2) # (Npoints, 1, D)
    c_j = F.normalize(c[None, :, :], p=2, dim=2) # (1, Nclusters, D)
    
    D_ij = torch.transpose((x_i * c_j).sum(-1), 0, 1) # (Nclusters, Npoints)

    values, indices = torch.topk(D_ij, 10) # (Nclusters, K)

    print(indices)
    #import pdb; pdb.set_trace()

    for idx, word_idx_list in enumerate(indices):
        aspect_word_list = 'aspect_%d: ' % (idx+1)
        for word_idx in word_idx_list:
            aspect_word_list += '%s, ' % word_dict[word_idx.item()]
        print(aspect_word_list)