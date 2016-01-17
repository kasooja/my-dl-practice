A simple RNN model is to predict the probability of a given sentence.
(Input Layer + Hidden Layer + Sofmax classification layer)

Input x is a sequence of words, and x_t is a one-hot vector of size: vocabulary size. 
Output o has a similar format, in which o_t is a vector of a vocabulary_size elementes,
and each element represents the probability of that word being the next word in the sentence.

Data: reddit comment data
To run: ./python train-rnn.py
 
