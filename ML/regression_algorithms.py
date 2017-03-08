# Chapter 12
# Spot-Check Regression Algorithms

# Regression Algorithms
## Linear ML Algorithms
### Linear Regression
### Ridge Regression
### LASSO Linear Regression
### Elastic Net Regression
## Nonlinear ML Algorithms
### k-Nearest Neighbors
### Classifcation and Regression Trees
### Support Vector Machines

import pandas
from sklearn import cross_validation
url = "https://goo.gl/sXleFv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
num_instances = len(X)
seed = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
scoring = 'neg_mean_squared_error'

# Linear Machine Learning Algorithms

## Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print (results.mean())

## Ridge Regression
### Extension of Linear Regression
### Loss function is modified to minimize the complexity of the model (sum squared value of coeff)
from sklearn.linear_model import Ridge
model = Ridge()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print (results.mean())

## LASSO Regression - Least Absolute Shrinkage and Selection Operator
### Modification of Linear regression: sum(abs(coefficient_values))
from sklearn.linear_model import Lasso
model = Lasso()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print (results.mean())

## ElasticNet Regression
### Regularization Regression (combines Ridge and Lasso)
from sklearn.linear_model import ElasticNet
model = ElasticNet()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print (results.mean())


# Nonlinear Machine Learning Algorithms

## K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print (results.mean())

## Classification and Regression Trees
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print (results.mean())

## Support Vector Machines (SVM) / Support Vector Regression (SVR)
from sklearn.svm import SVR
model = SVR()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print (results.mean())