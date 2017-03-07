# Ch 11
# Spot-Check Classification Algorithms

## Linear ML Algorithms
### Logistic Regression
### Linear Discriminant Analysis

## Nonlinear ML Algorithms
### k-Nearest Neighbors
### Naive Bayes
### Classification and Regression Trees
### Support Vector Machines

import pandas
from sklearn import cross_validation
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
num_instances = len(X)
seed = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

## Linear ML Algorithms


### Logistic Regression: binary classification problems
#### Assumes Gaussian Distribution for input values

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean()) # estimated accuracy

### Linear Discriminant Analysis (LDA): binary & multiclass
#### Assumes Gaussian Distribution for input values

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean())


## Nonlinear Machine Learning Algorithms

### K-Nearest Neighbors (KNN)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean())

### Naive Bayes
#### Gausian Distribution is assumed

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean())

### Classification and Regression Trees (CART or Decision Trees)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean())

### Support Vector Machines (SVM): line that best sperates the classes

from sklearn.svm import SVC
model = SVC()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean())