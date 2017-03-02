import theano
from theano import tensor

# two symbolic floating point scalars
a = tensor.dscalar()
b = tensor.dscalar()

# simple expression
c = a + b

# convert expression into a callable object that takes (a, b) and computes c
f = theano.function([a,b], c)

# bind 1.5 to a and 2.5 to b and evaluate c
result = f(1.5, 2.5)
print(result)
