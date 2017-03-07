# Chapter 9
# Evaluate the Performance of Machine Learning Algorithms with Resampling
# Resampling: Make accurate estimates for how well your algorithm will perform on new data

# Train and Test Sets; k-fold Cross Validation; Leave One Out Cross Validation; Repeated Random Test-Train Splits

import pandas
from sklearn.linear_model import LogisticRegression
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
model = LogisticRegression()

# Split into Train and Test Sets - simplest way to estimate perforamnce 

# 67% / 33% Split for Logistic Regression
from sklearn import cross_validation
test_size = .33
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print ("Accuracy: %.3f%%") % (result*100.0)

# K-fold Cross Validation
num_folds = 10
num_instances = len(X)

kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

# Leave One Out Cross Validation (k-fold cross validation when k=len(dataset) .. k is huge!)
num_folds = 10
num_instances = len(X)

loocv = cross_validation.LeaveOneOut(n=num_instances)
results = cross_validation.cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

# Repeated Random Test-Train Splits - Split data 67/33, randomly, 10 times
num_samples = 10
test_size = .33
num_instances = len(X)

kfold = cross_validation.ShuffleSplit(n=num_instances, n_iter=num_samples, test_size=test_size, random_state=seed)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
