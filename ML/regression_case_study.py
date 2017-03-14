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
plt.show()

### density plots
print ('density plots')
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1)
plt.show()

### box and whisker plots
print ('box and whisker plots')
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)