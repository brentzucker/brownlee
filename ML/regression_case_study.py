# Chapter 20
# Regression Machine Learning Case Study Project

# Regression Predictive Modeling Problem end to end
# Data Transforms to Improve Model Performace
# Algorithm Tuning to improve Model Performance
# Ensemble Methods to improve model performance

# Problem Definition

# Boston House Price Dataset
# Attributes
# 1. CRIM: per capita crime rate by town
# 2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS: proportion of non-retail business acres per town
# 4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5. NOX: nitric oxides concentration (parts per 10 million)
# 6. RM: average number of rooms per dwelling
# 7. AGE: proportion of owner-occupied units built prior to 1940
# 8. DIS: weighted distances to five Boston employment centers
# 9. RAD: index of accessibility to radial highways
# 10. TAX: full-value property-tax rate per $10,000
# 11. PTRATIO: pupil-teacher ratio by town
# 12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13. LSTAT: % lower status of the population
# 14. MEDV: Median value of owner-occupied homes in $1000s

# Load libraries
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
url = "https://goo.gl/sXleFv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pandas.read_csv(url, delim_whitespace=True, names=names)

# Analyze Data

## Descriptive Statistics
print ('data shape ' + str(dataset.shape))

## Data types
print ('\ndata types\n' + str(dataset.dtypes))

## head
print ('\nhead\n' + str(dataset.head(20)))

## descriptions
pandas.set_option('precision', 1)
print ('\ndescriptions\n' + str(dataset.describe()))

## correlation
pandas.set_option('precision', 2)
print ('\ncorrelation\n' + str(dataset.corr(method='pearson')))

# Data Visualizations
print ('\nData Visualizations\n')

## Unimodel Data Visualizations

### histograms
print ('histograms')
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
# plt.show()

### density plots
print ('density plots')
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1)
# plt.show()

### box and whisker plots
print ('box and whisker plots')
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
# plt.show()

## Multimodel Data Visualizations

### scater plot matrix
print ('scatter plot matrix')
scatter_matrix(dataset)
# plt.show()

### correlation matrix
print ('correlation matrix')
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none') 
fig.colorbar(cax) 
ticks = np.arange(0,14,1)
ax.set_xticks(ticks) 
ax.set_yticks(ticks) 
ax.set_xticklabels(names) 
ax.set_yticklabels(names)
# plt.show()

## summary of ideas
### feature selection and removing the most correlated attributes
### normalizing the dataset to reduce the effect of different scales
### standardizing the dataset to reduce the effects of differing distributions
### binning (discretization) of the data (improve accuracy for decision tree algorithms)

# split-out Validation Dataset
values = dataset.values
X = values[:,0:13]
Y = values[:,13]
validation_size = .2
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Evaluate Algorithms: Baseline

# Evaluation Metric
# Test options and evaluation metric 
num_folds = 10 
num_instances = len(X_train) 
seed = 7 
scoring = 'neg_mean_squared_error'

## Linear Algorithms: Linear Regression (LR); Lasso Regression (LASSO); ElasticNet (EN)
## Nonlinear Algorithms: Classification and Regression Trees (CART); Support Vector Regression (SVR); k-Nearest Neighbors (KNN)

# Spot-Check Algorithms
print ('\nSpot-Check Algorithms\n')
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
print ('\nCompare Algorithms\n')
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
# plt.show()

# Evaluate Algorithms: Standardization
## Transform data so that each attribute has a mean value of zero and a standard deviation of 1
## Use pipelines to avoid data leakage

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Standardized Algorithms 
fig = plt.figure() 
fig.suptitle('Scaled Algorithm Comparison') 
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names)
# plt.show()

# Improve Results with Tuning
## KNN default numNeighbors is 7 .. use grid search to try a set of different numbers (1-21) of neighbors and see if we can improve the score
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

# Ensemble Methods to improve performance
## Boosting Methods: AdaBoost(AB), Gradient Boosting (GBM)
## Bagging Methods: Random Forests (RF), Extra Trees (ET)

# Evaluate ensemble methods
print ('\nEvaluate Ensemble Methods\n')
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Ensemble Algorithms
fig = plt.figure() 
fig.suptitle('Scaled Ensemble Algorithm Comparison') 
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names) 
# plt.show()

# Tune Ensemble Methods
## Default numOfBoostingStates to perform (n_estimators) is 100 for Gradient Boosting (GB)
## The more estimators, the longer the training time
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50, 100, 150, 200, 250, 300, 350, 400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

# Finalize Model

print ('\nFinalize Model\n')
## Prepare model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)

## Scale inputs for the validation set and generate predictions
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print ('Mean Squared Error ' + str(mean_squared_error(Y_validation, predictions)))