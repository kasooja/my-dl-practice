# ## Read data + train the network
# Finally we have all blocks to create a MLP for the MNIST dataset.
# 
# You find the MNIST dataset in the data dir. Otherwise you can obtain it from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

# In[ ]:

import pickle
import gzip
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from Lec2.org.insight.digitrecognizer.demo.MLP import MLP



# Load the pickle file for the MNIST dataset.
dataset = '/Users/kat/PycharmProjects/DeepLearningLecs/data/mnist.pkl.gz'

f = gzip.open(dataset, 'rb')
train_set, dev_set, test_set = pickle.load(f, encoding='latin1')
f.close()

#train_set contains 2 entries, first the X values, second the Y values
train_x, train_y = train_set
dev_x, dev_y = dev_set
test_x, test_y = test_set

#Created shared variables for these sets (for performance reasons)
train_x_shared = theano.shared(value=np.asarray(train_x, dtype='float32'), name='train_x')
train_y_shared = theano.shared(value=np.asarray(train_y, dtype='int32'), name='train_y')


print("Shape of train_x-Matrix: ",train_x_shared.get_value().shape)
print("Shape of train_y-vector: ",train_y_shared.get_value().shape)
print( "Shape of dev_x-Matrix: ",dev_x.shape)
print ("Shape of test_x-Matrix: ",test_x.shape)

###########################
#
# Start to build the model
#
###########################

# Hyper parameters
hidden_units = 50
learning_rate = 0.01
batch_size = 20

# Variables for our network
index = T.lscalar()  # index to a minibatch
x = T.fmatrix('x')  # the data, one image per row
y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

rng = np.random.RandomState(1234) #To have deterministic results

# construct the MLP class
classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=50, n_out=10)

# Define our cost function = error function
cost = classifier.negative_log_likelihood(y) #Here we could add L1 and L2 terms for regularization

# Update param := param - learning_rate * gradient(cost, param)
# See Lecture 1 slide 28
updates = [(param, param - learning_rate * T.grad(cost, param) ) for param in classifier.params]

# Now create a train function
# The train function needs the data, the index for the minibatch and the updates to work correctly
train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x_shared[index * batch_size: (index + 1) * batch_size],
            y: train_y_shared[index * batch_size: (index + 1) * batch_size]
        }
    )

# Create a prediction function
predict_labels = theano.function(inputs=[x], outputs=classifier.softmaxLayer.y_pred)

print (">> train- and predict-functions are compiled <<")


# **Time to train the model**
# 
# Now we can train our model by calling train_model(mini_batch_index). To predict labels, we can use the function predict_labels(data).

# In[ ]:

number_of_minibatches = len(train_x) / batch_size

print ("%d mini batches" % (number_of_minibatches))

number_of_epochs = 10
print ("%d epochs" % number_of_epochs)


#
def compute_accurarcy(dataset_x, dataset_y):
    predictions = predict_labels(dataset_x)
    errors = sum(predictions != dataset_y)  # Number of errors
    accurarcy = 1 - errors / float(len(dataset_y))
    return accurarcy


for epoch in range(number_of_epochs):
    # Train the model on all mini batches
    for idx in range(0, int(number_of_minibatches)):
        train_model(idx)

    accurarcy_dev = compute_accurarcy(dev_x, dev_y)
    accurarcy_test = compute_accurarcy(test_x, test_y)

    print(
    "%d epoch: Accurarcy on dev: %f, accurarcy on test: %f" % (epoch, accurarcy_dev, accurarcy_test))

print("DONE")