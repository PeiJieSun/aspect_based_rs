import time

import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from gensim.models import Word2Vec
from nltk.corpus import stopwords 

from collections import defaultdict

import config as conf

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

def load_glove_embeddings():
    glove_embedding_path = '/content/drive/My Drive/datasets/glove/glove.6B.300d.txt'
    embeddings_dict = {}
    with open(glove_embedding_path, 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

if __name__ == "__main__":
    vocab = np.load('%s/%s.vocab.npy' % (conf.target_path, conf.data_name), allow_pickle=True).item()
    vocab_decoder = \
        np.load('%s/%s.vocab_decoder.npy' % (conf.target_path, conf.data_name), allow_pickle=True).item()
    
    word_id_list = list(vocab_decoder.keys())
    word_id_list.sort()

    glove_embedding_dict = load_glove_embeddings()
    word_embedding_dict = {}
    #import pdb; pdb.set_trace()

    x = []
    for word_id in word_id_list:
        word = vocab_decoder[word_id]
        if word in glove_embedding_dict:
            x.append(glove_embedding_dict[word])
            word_embedding_dict[word_id] = glove_embedding_dict[word]
        else:
            x.append(np.zeros(300))
            word_embedding_dict[word_id] = np.zeros(300)
    np.save('%s/%s.word_embedding' % (conf.target_path, conf.data_name), word_embedding_dict)

    x = torch.FloatTensor(x).cuda() # (Npoints, D)

    # get the cluster center vectors c 
    K = 15
    clx, c = KMeans(x, K)
    check_dir('%s/%s.k_means_15' % (conf.target_path, conf.data_name))
    np.save('%s/%s.k_means_15' % (conf.target_path, conf.data_name), tensorToScalar(c))
    
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
            aspect_word_list += '%s, ' % vocab_decoder[word_idx.item()]
        print(aspect_word_list)