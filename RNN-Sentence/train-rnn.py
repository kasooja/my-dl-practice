#! /usr/bin/env python
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from rnn_sentence import SimpleRNN
import data_helpers


print "------------ Loading data -------------"
# set vocabulary size
vocabulary_size = 100
X_train, Y_train, Word_Dict = data_helpers.create_training_data(vocabulary_size)


# model initialization with size of hidden layer
model = SimpleRNN(vocabulary_size, hidden_dim=20)


print "------------ Traning model --------------" 
# set initital availables
learning_rate=0.005
nEpoch = 1
losses = []
num_instances_seen = 0

t1 = time.time()
model.sgd_step(X_train[10], Y_train[10], learning_rate=0.005)
t2 = time.time()
print "Applying SGD for the first time: %f milliseconds" % ((t2 - t1) * 1000.)

for epoch in range(nEpoch):
	loss = model.calculate_loss(X_train, Y_train)
	losses.append((num_instances_seen, loss))
	time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	print "%s: Loss after seeing %d examples vs epoch=%d: %f" % (time, num_instances_seen, epoch, loss)
    # Adjust the learning rate if loss increases
	if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
        	learning_rate = learning_rate * 0.5  
		print "Setting learning rate to %f" % learning_rate
    	sys.stdout.flush()
    	# For each training example...
    	for i in range(len(Y_train)):
            	# One SGD step
        	model.sgd_step(X_train[i], Y_train[i], learning_rate)
        	num_instances_seen += 1

print "---------- Testing ------------" 
predictions = model.predict(X_train[100])
print X_train[100]
print '***********'
print predictions.shape
print '***********'
print predictions


