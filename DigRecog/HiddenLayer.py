import numpy
import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        :param rng: Random number generator, for reproducable results
        :param input: Symbolic Theano variable for the input
        :param n_in: Number of incoming units
        :param n_out: Number of outgoing units
        :param W: Weight matrix
        :param b: Bias
        :param activation: Activation function to use
        """
        self.input = input
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if W is None:  # Initialize Glorot Style
            W_values = numpy.asarray(
                rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)

            if activation == theano.tensor.nnet.sigmoid or activation == theano.tensor.nnet.hard_sigmoid or activation == theano.tensor.nnet.ultra_fast_sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W')

        if b is None:  # Initialize bias to zeor
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        # Compute the activation
        lin_output = T.dot(input, self.W) + self.b

        # Compute the output
        if activation is None:
            self.output = lin_output
        else:
            self.output = activation(lin_output)

        # Parameters of the model that can be trained
        self.params = [self.W, self.b]