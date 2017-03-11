# Chapter 15
# Improve Performace with Ensemble Predictions

# Ensembles increase accuracy by combining multiple models

# Types of Ensemble Predictions
## Bagging: typically same type of model but different subsamples of traing set
## Boosting: typically same type of model but each learns to fix prediction error of prior model
## Voting: typically different types of models, uses simple stats (like mean) to combine predictions

import pandas as pd
from sklearn import cross_validation
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
num_instances = len(X)
seed = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

# Bagging: Bootstrap Aggregation
## take multiple samples from training set (with replacement) and train a model for each sample
## final output is predicted average across all predictions of each submodel
### Bagged Decision Trees
### Random Forest
### Extra Trees

from sklearn.ensemble import BaggingClassifier

## Bagged Decision Trees
### Bagging is good for algorithms with high variance

from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean())

## Random Forest: random subset of features rather than greedy
from sklearn.ensemble import RandomForestClassifier
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean())

## Extra Trees: random trees are constructed from the training set
from sklearn.ensemble import ExtraTreesClassifier
num_trees = 100
max_features = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print (results.mean())


# Boosting Algorithms
## sequence of models to attempt to correct the mistakes of the models before them
## models make predictions which may be weighted depending on their accuracy
## Results are combined to create a final output prediction
### Adaboost
### Stochastic Gradient Boosting

## Adaboost: weights instances of the dataset by how easy/difficult they are to classify which alllows the algorithm to pay
### more/less attention to them when constructing a model

### 30 decision trees using Adaboost algorithm
from sklearn.ensemble import AdaBoostClassifier
num_trees = 30
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

## Stochastic Gradient Boosting - Gradient Boosting Machines
from sklearn.ensemble import GradientBoostingClassifier
num_trees = 100
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# Voting Ensemble: wraps 2 or more models and averages the prediction
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC()

estimators = []
estimators.append(('logistic', model1))
estimators.append(('cart', model2))
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_validation.cross_val_score(ensemble, X, Y, cv=kfold)
print (results.mean())