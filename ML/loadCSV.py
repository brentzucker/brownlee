# Ch4 pg 27
import csv
import numpy as np
filename = 'C:\\Users\\bz185013\\Documents\\Datasets\\pima-indians\\pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rb')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')
print(data.shape)

### Numpy
data = np.loadtxt(raw_data, delimiter=',')
print(data.shape)

# Import using numpy from url
import urllib
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
raw_data = urllib.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=',')
print(dataset.shape)

# pandas
import pandas
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
data = pandas.read_csv(filename, names=names)
print(data.shape)

data = pandas.read_csv(url, names=names)
print(data.shape)
