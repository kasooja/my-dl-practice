import numpy
import theano
import theano.tensor as T


class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX), name='W')
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX), name='b')

        # Compute the output of the softmax layer, we call it P(y | x), i.e. how
        # likely is the label y given the input x
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # For prediction we select the most probable output
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """
        Computes the negative log-likelihood. The function explained:

        T.log(self.p_y_given_x): Compute the negative log-likelihood of p_y_given_x
        T.arange(y.shape[0]), y]: Select the neuron at position y, our label
        T.mean(): Compute the average over our mini batch
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])