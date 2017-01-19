import theano
import theano.tensor as T
import numpy as np
#Define my internal state
init_value = 1
state = theano.shared(value=init_value, name='state')
#Define my operation f(x) = 2*x on the internal state variable shared
x = T.lscalar('x')
z = 2*x
accumulator = theano.function(inputs=[], outputs=z, givens={x: state})
print(accumulator())
print(accumulator())
