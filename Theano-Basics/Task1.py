import theano
import theano.tensor as T
x = T.dscalar('x') #First input variable to the compute graph
y = T.dscalar('y') #Second input variable to the compute graph
z = 3*x + x*y + 3*y #Our formula we like to compute
#Compile for the output z, given the inputs x and y
f = theano.function(inputs=[x,y], outputs=z)
# Now you can invoke f and pass the input values, i.e. f(1,1), f(10,-3) and the result for this operation is returned.
# In[9]:
print(f(1,1))
print(f(10,-3))
