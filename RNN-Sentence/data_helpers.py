#!/usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
	
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "START_SENTENCE"
sentence_end_token = "END_SENTENCE"

def create_training_data(vocab_size):
	with open('data/reddit-comments.csv', 'rb') as f:
    		reader = csv.reader(f, skipinitialspace=True)
    		reader.next()
    		# Sentences tokenization
    		sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    		# adding START_SENTENCE and END_SENTENCE
    		sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

	print "The number of sentences: %d" % (len(sentences))

	# Work tokenization
	tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

	# Compute word frequencies
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	print "There are %d unique words tokens." % len(word_freq.items())

	# Get the most common words and build index_to_word and word_to_index vectors
	vocab = word_freq.most_common(vocab_size-1)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token)
	word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
	
	#print word_to_index

	print "Vocabulary size: %d." %vocab_size

	# Replace all words not in the vocabulary with UNKNOWN_TOKEN
	for i, sent in enumerate(tokenized_sentences):
    		tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

	# Create the training data
	X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
	Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
	
	return X_train, Y_train, word_to_index


