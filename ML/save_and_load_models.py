# Ch 17
# Save and Load ML Models

# pickle and joblib

# pickle: serializing standard python objects
# Example demonstrates train a logistic regression model, save the model and load it to make predictions on the unseen test

import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import pickle
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

# predict 1 test
rowNum = 0
input = X_test[rowNum,:]
knownOutput = Y_test[rowNum]
output = loaded_model.predict(input)
print input
print str(knownOutput) + ' ?= ' + str(output)


# Joblib
# joblib: efficient way to store numpy data structures.. some ML algos require a lot of parameters or store the entire dataset (knn)
from sklearn.externals import joblib

# save the model to disk
joblib.dump(model, filename)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)