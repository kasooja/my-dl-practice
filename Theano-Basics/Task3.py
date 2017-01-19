import theano
import theano.tensor as T
import numpy as np
x = T.fvector('x')
W = T.fmatrix('W')
b = T.fvector('b')
activation = T.dot(x,W)+b
z = T.tanh(activation)
f = theano.function(inputs=[x,W,b], outputs=[activation,z])
# Next we define some NumPy-Array with data and let Theano compute the result for $f(x,W,b)$
# In[12]:
inputX = np.asarray([0.1, 0.2, 0.3], dtype='float32')
inputW = np.asarray([[0.1,-0.2],[-0.4,0.5],[0.6,-0.7]], dtype='float32')
inputB = np.asarray([0.1,0.2], dtype='float32')
print("inputX.shape",inputX.shape)
print("inputW.shape",inputW.shape)
f(inputX, inputW, inputB)
