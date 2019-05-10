'''
word2vec enriched with point-wise mutual information
version: 0.2
author: Michal Pikusa (pikusa.michal@gmail.com)
'''


import time
import sys
import pandas as pd
import collections
import numpy as np
import pickle
from tqdm import tqdm
import math
import random
import itertools
from numpy import inf
import h5py
from multiprocessing.dummy import Pool as ThreadPool
from sklearn import random_projection
from nltk.util import skipgrams
import ahocorasick
import string

def main():
    # Start the counter
    start = time.time()

    # Load raw data and tokenize
    corpus_file = str(sys.argv[1])
    print("Processing...")
    merged = []
    no_words = 0
    window_size = 2
    skipgram_list = []
    def read_words(inputfile):
        with open(inputfile, 'r',encoding='utf-8') as f:
            while True:
                buf = f.read(102400)
                if not buf:
                    break
                while not str.isspace(buf[-1]):
                    ch = f.read(1)
                    if not ch:
                        break
                    buf += ch
                words = buf.split()
                for word in words:
                    yield word
            yield ''
    no_lines = 0
    
    line_tokens = []
    for word in read_words(corpus_file):
        no_lines += 1
        if '.' not in word:
            line_tokens.append(word.translate(str.maketrans('', '', string.punctuation)))
        else:
            line_tokens.append(word.translate(str.maketrans('', '', string.punctuation)))
            merged.append(line_tokens)
            line_tokens = []
        if no_lines % 100000 == 0:
        	sys.stdout.write("\rWords: %i" % int(no_lines))
        	sys.stdout.flush()

    # Function to create a dataframe with counts and probabilities
    def create_count_df(list_to_count,skipgrams,sample_rate):
        list_with_counts = collections.Counter(list_to_count)
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        if skipgrams==False:
            df['word'] = list_with_counts.keys()
            df['count'] = list_with_counts.values()
            df = df[df['count'] > 4]
            df['prob'] = df['count'] / sum(df['count'])
            # calculate negative sample probability
            df['weight'] = df['prob'] ** (3/4)
            df['neg_samp_prob'] = df['weight'] / sum(df['weight'])
            # subsample
            df['prob_sub'] = (np.sqrt(df['prob']/sample_rate)+1)*sample_rate/df['prob']
            df2['word'] = list_to_count
            df2 = df2.join(df.set_index('word'), on='word')
            df = df[df['prob_sub'] > sample_rate]
            df2 = df2[df2['count'] > 4]
            df2 = df2[df2['prob_sub'] > sample_rate]
            df = df[['word','prob','neg_samp_prob']]
            return df, df2['word'].tolist()
        else:
            word_list1 = []
            word_list2 = []
            for item in list_with_counts.keys():
                word_list1.append(item[0])
                word_list2.append(item[1])
            df['word1'] = word_list1
            del word_list1
            df['word2'] = word_list2
            del word_list2
            df['count'] = list_with_counts.values()
            df['prob'] = df['count'] / sum(df['count'])
            df = df[['word1','word2','prob']]
            return df

    # Create the list of unigrams with the count and normalize probability
    print("\nCreating the list of unigrams...")
    sample_rate = 0.001
    unigram_df, tokens = create_count_df([item for sublist in merged for item in sublist],False,sample_rate)
    print("# unigrams: ", unigram_df.shape[0])
    print("Creating the list of skipgrams...")
    no_words = 0
    A = ahocorasick.Automaton()
    for idx, key in enumerate(tokens):
        A.add_word(key, (idx, key))
    for line_counter,line in enumerate(merged):
        line_skipgrams = list(skipgrams(line, window_size, window_size))
        for skipgram in line_skipgrams:
            if (skipgram[0] in A) and (skipgram[1] in A):
                skipgram_list.append(skipgram)
        if no_lines % 100000 == 0:
            sys.stdout.write("\rWords: %i" % int(no_words))
            sys.stdout.flush()
            no_words += len(line)
    del tokens
    del merged
    print("\nCreating skipgram data frame...")
    skipgram_df = create_count_df(skipgram_list,True,sample_rate)
    print("# skipgrams: ", skipgram_df.shape[0])
    del skipgram_list
    
    # Optimize the skipgram dataframe to reduce the size by ~ 90%
    print("Optimizing...")
    skipgram_df['word1'] = skipgram_df.word1.astype('category')
    skipgram_df['word2'] = skipgram_df.word2.astype('category')
    skipgram_df['prob'] = skipgram_df.prob.astype('float32')

    # Calculate PMI values for each skipgram
    print("Calculating PMI...")
    unigram_df = unigram_df.set_index('word')
    skipgram_df['prob1'] = skipgram_df['word1'].map(unigram_df['prob'].get).astype('float32')
    skipgram_df['prob2'] = skipgram_df['word2'].map(unigram_df['prob'].get).astype('float32')
    skipgram_df['pmi'] = np.log(skipgram_df['prob']/(skipgram_df['prob1']*skipgram_df['prob2'])).astype('float32')
    skipgram_df = skipgram_df[['word1','word2','pmi']]

    unigram_df = unigram_df.reset_index()
    vocab_length = unigram_df.shape[0]

    # Create the unigram table for negative sampling
    print("Creating negative samples table...")
    table_size = 10000000
    neg_samp_list = []
    row_index = 0
    for index,row in unigram_df.iterrows():
        rate = row['neg_samp_prob'] * table_size
        for i in range(0,int(rate)):
            neg_samp_list.append(row_index)
        row_index += 1
    
    # Create the list of co-occurence probabilities for the output vector
    print("Preparing empty output vectos...")
    output_vectors = np.zeros(shape=(vocab_length,vocab_length))
    unigram_index_list = list(range(0,vocab_length))
    unigram_df['index_list'] = unigram_index_list
    del unigram_index_list
    unigram_df = unigram_df.set_index('word')
    skipgram_df['word_index1'] = skipgram_df['word1'].map(unigram_df['index_list'].get)
    skipgram_df['word_index2'] = skipgram_df['word2'].map(unigram_df['index_list'].get)
    skipgram_df = skipgram_df[['word_index1','word_index2','pmi',]]
    print("Populating the output vectors for training...")
    row_count = 0
    for row in skipgram_df.itertuples(index=True):
        output_vectors[int(getattr(row, "word_index1"))][int(getattr(row, "word_index2"))] = getattr(row, "pmi")
        if row_count % 1000000 == 0:
        	sys.stdout.write("\rProgress: %.2f %%" % float(row_count/skipgram_df.shape[0])*100)
        	sys.stdout.flush()
        row_count += 1
    unigram_df = unigram_df.reset_index()
    word_list = unigram_df['word'].tolist()
    del unigram_df

    #  Initialize the weights
    print("\nInitializing the network...")
    no_hid = int(sys.argv[2])
    epochs = 5
    neg_samp_no = 20
    starting_alpha = 0.025
    alpha = starting_alpha
    syn0 = np.random.uniform(low=-0.5/no_hid, high=0.5/no_hid, size=(vocab_length, no_hid))
    syn1 = np.zeros(shape=(vocab_length, no_hid))

    # Helper functions
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # Train the network using negative sampling
    print("\nTraining the network...")
    train_counter = 0
    for epoch in range(0, epochs):
        for row in skipgram_df.itertuples(index=True):
            u_layer = np.zeros(no_hid)
            softmax_array = []
            c_index_array = []
            w_index = int(getattr(row, "word_index1"))
            c_index_array.append(int(getattr(row, "word_index2")))
            softmax_array.append(float(getattr(row, "pmi")))
            for neg_samp in range(0,neg_samp_no):
                neg_samp_pos = random.randint(0,len(neg_samp_list)-1)
                c_index = neg_samp_list[neg_samp_pos]
                c_index_array.append(c_index)
                softmax_array.append(output_vectors[w_index][c_index])
            softmax_array = softmax(softmax_array)
            for i in range(0,len(c_index_array)):
                c_index = c_index_array[i]
                label = softmax_array[i]
                if alpha < starting_alpha * 0.0001:
                    alpha = starting_alpha
                f1 = sigmoid(np.dot(syn0[w_index], syn1[c_index]))
                f1_error = alpha * (label - f1)
                u_layer += f1_error * syn1[c_index]
                syn1[c_index] += f1_error * syn0[w_index]
                alpha -=  starting_alpha/(skipgram_df.shape[0]*epochs*neg_samp_no)
            syn0[w_index] += u_layer
            if train_counter % 10000 == 0:
                sys.stdout.write("\rProgress: %.2f %% Alpha: %.5f" % (float(train_counter/(skipgram_df.shape[0]*epochs))*100,float(alpha)))
                sys.stdout.flush()
            train_counter += 1

    # Save the model
    print("\nSaving the model...")
    word_list_name = '_'.join([sys.argv[3], 'wordlist.p'])
    vectors_name = '_'.join([sys.argv[3], 'vectors.hdf5'])
    output_word_list = open(word_list_name, 'wb')
    pickle.dump(word_list, output_word_list)
    output_word_list.close()
    vectors_file = h5py.File(vectors_name, 'w')
    vectors_file.create_dataset('vectors', data=syn0)
    vectors_file.close()

    # Print out overall statistics of the run
    end = time.time()
    print("Running time: ", str(int((end - start)/60)), "minutes")

if len(sys.argv) < 4:
    print("Missing arguments!")
    print("Correct syntax: python word2vec.py <text_file> <number_of_vectors> <model_name>")
    print("Example: ")
    print("python create_vectors.py wiki.txt 100 wiki")
else:
    main()