# Ch 7
# Prepare your data for Machine Learning

import pandas
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = pandas.read_csv(url, names=names)
array = dataframe.values

# seperate array into input and output components
x = array[:, 0:8]
y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(x)

# Pre processing - Data transformations

# Fit and Multiple Transform
# fit(): prepare parameters of transform on your data
# transform(): to prepare for modeling

# Rescale Data 
# Normalization

# summarize transformed data
np.set_printoptions(precision=3)
# print (rescaledX[0:5, :])

# Standardize Data
# transform attribute swith a gaussian distribution and differing means and standard deviations to standard gaussian distribution
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x)
rescaledX = scaler.transform(x)

# summarize transformed data
# print (rescaledX[0:5,:])

# Normalize Data: rescaled each row have a length of 1 (called a unit norm or a vector with a length of 1)
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(x)
normalizedX = scaler.transform(x)
# print (normalizedX[0:5,:])

# Binarize Data (make binary)
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(x)
binaryX = binarizer.transform(x)
print(binaryX[0:5,:])
