from Lec2.org.insight.digitrecognizer.demo.HiddenLayer import HiddenLayer
from Lec2.org.insight.digitrecognizer.demo.SoftmaxLayer import SoftmaxLayer
import theano.tensor as T

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """
        :param rng: Our random number generator
        :param input: Input variable (the data)
        :param n_in: Input dimension
        :param n_hidden: Hidden size
        :param n_out: Output size
        """
        self.hiddenLayer = HiddenLayer(rng=rng,
                                       input=input, n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        self.softmaxLayer = SoftmaxLayer(
            input=self.hiddenLayer.output,
            n_in=n_hidden, n_out=n_out)

        # Negative log likelihood of this MLP = neg. log likelihood of softmax layer
        self.negative_log_likelihood = self.softmaxLayer.negative_log_likelihood

        # Parameters of this MLP = Parameters offen Hidden + SoftmaxLayer
        self.params = self.hiddenLayer.params + self.softmaxLayer.params



        # ## Read data + train the network


# Finally we have all blocks to create a MLP for the MNIST dataset.
