'''
word2vec enriched with point-wise mutual information - script for finding nearest neighbors (similar words)
Description: Script for finding similar words - it uses clusters from cluster.py, so make sure you run it first after training the model
with word2vec.py
version: 0.2
author: Michal Pikusa (pikusa.michal@gmail.com)
'''


import time
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import h5py
import numpy as np

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
    with open('clusters.p', 'rb') as cl:
        clusters = pickle.load(cl)

    # Get the data
    vectors = f['vectors'][()]
    word = sys.argv[2]

    # Convert to a data frame
    clustered = pd.DataFrame(clusters, columns=['cluster'])
    clustered['vectors'] = vectors.tolist()
    clustered['word'] = word_list
    print(clustered.head(5))
    # Find appropriate word index
    for i, j in enumerate(word_list):
        if j == word:
            word_index = i
    sort_by_cluster = int(clustered.iloc[[word_index]]['cluster'])
    clustered = clustered[clustered['cluster'] == sort_by_cluster]
    clustered = clustered.reset_index()
    word_list = clustered['word'].tolist()
    for i, j in enumerate(word_list):
        if j == word:
            word_index = i
    print(word_index)
    vectors = clustered['vectors'].tolist()

    # Calculate cosine similarity with all other words
    similar_values = []
    similar_words = []
    for i, j in enumerate(vectors):
        similarity = (cosine_similarity(np.array(vectors[word_index]).reshape(1,-1), np.array(vectors[i]).reshape(1,-1)))[0][0]
        similar_values.append(abs(similarity))
        similar_words.append(word_list[i])

    # Create a dataframe with all similarities and word indices, sort it and get the most similar words
    similarities = pd.DataFrame()
    similarities['word'] = similar_words
    similarities['similarity'] = similar_values
    similarities = similarities.sort_values(by=similarities.columns[1], ascending=False)
    print(similarities.head(n=10))

    # Print out overall statistics of the run
    end = time.time()
    print("Running time:", str(round(end - start, 1)),"seconds")
    return

if len(sys.argv) < 3:
    print("Missing arguments!")
    print("Correct syntax: python find_nn.py <model_name> <word>")
    print("Example: ")
    print("python find_nn.py wiki poland")
else:
    main()
