# Ch 19
# Your First Machine Learning Project in Python Step-By-Step

# Iris Flowers
# 1. Load Dataset
# 2. Summarize the Data
# 3. Visualize the Data
# 4. Evaluate some algorithms
# 5. Make some predictions

# Load libraries
import pandas
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Dataset
url = "https://goo.gl/mLmoIz"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Summarize The Data
print ('Summary of Data')

# Data Dimensions
print ('Dimensions ' + str(dataset.shape))

# Peek at the data
print ('Peek at first 20 rows')
print (dataset.head(20))

# Statistical Summary
print ('Statistical Summary')
print (dataset.describe())

# Class Distribution
print ('Class Distribution')
print (dataset.groupby('class').size())

# Data Visualization (uni/multi variate plots)
print ('\nData Visualization')

# Univariate Plots: help better understand each attribute

## Box and Whisker
print ('Box and Whisker')
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

## Histograms
print ('Histograms')
dataset.hist()
plt.show()

# Multivariate plots help better understand relationships between attributes

## Scatter Plot Matrix
print ('Scatter Plot Matrix')
scatter_matrix(dataset)
plt.show()

# Evaluate Some Algorithms

## Split-out Validation Datset
values = dataset.values
X = values[:, 0:4]
Y = values[:, 4]
validation_size = .2
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

## Test Harness for 10-fold cross validation
num_folds = 10
num_instances = len(X_train)
seed = 7
evalution_metric_scoring = 'accuracy' # numCorrectPredictedInstances / totalNumInstances

## Build Models
### Simple Linear
#### Logisitic Regression (LR)
#### Linear Discriminant Analysis (LDA)
### Nonlinear
#### k-Nearest Neighbors (KNN)
#### Classification and Regression Trees (CART)
#### Gaussian Naive Bayes (NB)
#### Support Vector Machines (SVM)

# Spot-Check Algorithms
print ('\nSpot-Check Algorithms')
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN' , KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=evalution_metric_scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print (msg)

# Select the Best Model .. KNN has the largest estimated accuracy score

# Compare algorithms
print('\nAlgorithms Comparison Visualization')
fig = plt.figure()
fig.suptitle('Algorithms Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make Predictions
print ('\nMake Predictions')
## KNN has the highest accuracy
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print ('accuracy score ' + str(accuracy_score(Y_validation, predictions)))
print ('confusion matrix')
print (confusion_matrix(Y_validation, predictions))
print ('classification report')
print (classification_report(Y_validation, predictions))