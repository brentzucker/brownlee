# Ch 16
# Improve Performance with Algorithm Tuning

# Grid search and random search algorithm tuning strategy

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import Ridge
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
model = Ridge()

# Grid Search

from sklearn.grid_search import GridSearchCV
alphas = np.array([1,0.1,0.01,0.001,0.0001,0]) # Different alpha values for ridge regression
param_grid = dict(alpha=alphas)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

# Random Search Parameter Tuning
## sample algorithm parameters from a random distribution

## random alpha values between 0 and 1
from scipy.stats import uniform
from sklearn.grid_search import RandomizedSearchCV
seed = 7

param_grid = {'alpha': uniform()}
iterations = 100
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=iterations, random_state=seed)
rsearch.fit(X, Y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)