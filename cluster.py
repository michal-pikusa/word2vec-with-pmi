'''
word2vec enriched with point-wise mutual information - clustering script
Description: Use it after training the model with word2vec.py to cluster the result vectors
             it will speed up finding nearest neighbors with find_nn.py
version: 0.2
Author: Michal Pikusa (pikusa.michal@gmail.com)
'''

import time
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import h5py
from nltk.cluster.kmeans import KMeansClusterer
import nltk
from sklearn.cluster import KMeans

def main():
    # Start the counter
    start = time.time()

    # Load data
    print("Loading data...")
    word_list_name = '_'.join([sys.argv[1], 'wordlist.p'])
    vectors_name = '_'.join([sys.argv[1], 'vectors.hdf5'])
    with open(word_list_name, 'rb') as wl:
        word_list = pickle.load(wl)
    f = h5py.File(vectors_name, 'r')
    a_group_key = list(f.keys())[0]
    # Get the data
    vectors = f[a_group_key]
    end = time.time()
    print(end-start)
    print("clustering")
    kmeans = KMeans(n_clusters=50, random_state=0).fit(vectors)
    assigned_clusters = kmeans.labels_
    end = time.time()
    output_cluster_list = open('clusters.p', 'wb')
    pickle.dump(assigned_clusters, output_cluster_list)
    output_cluster_list.close()
    print(end-start)

if len(sys.argv) < 2:
    print("Missing arguments!")
    print("Correct syntax: python cluster.py <model_name>")
    print("Example: ")
    print("python test_mp.py wiki")
else:
    main()
