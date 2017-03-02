from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load pima indians dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split dataset into input variables (x) [first 8 values], and output variables (y) [last variable]
x = dataset[:, 0:8]
y = dataset[:, 8]

# create model: Sequential Model
# make sure first layer has right number of inputs: specified by input_dim (should be same value as input variables)
# Dense Class: Fully connected layers
# Dense(numberOfNeuronsInLayer, initializationMethod, activationFunction)
# uniform initialization: small random number generated from uniform distribution (b/w 0-.05)
# normal initialization: small random numbers from Gaussian distribution
# activation functions = rectifier (relu).. good performance and sigmoid [output b/w 0 and 1]
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
# logarithmic loss function to evaluate set of weights (binary classification requires specific logarithmic loss function)
# optimizer used to search through different weights for the network (gradient descent algorithm: adam - random efficient algorithm)
# optional metrics to collect and report during training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
# epochs: number of iterations the training process runs
# batch size: number of instances that are evaluated before a weight update in the network
# small number of epochs and batch size is chosen... these can be chosen experimentally with trial and error
model.fit(x, y, nb_epoch=150, batch_size=10)

# Evaluate Model
# how well we modeled the dataset, not how it will perform on new data
scores = model.evaluate(x,y)
# See a message for each epoch 
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
