# Ch 10
# Machine Learning Algorithm Performance Metrics

## Classification Metrics

import pandas
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
num_instances = len(X)
seed = 7
model = LogisticRegression()

### Classification Accuracy: correct predictions made as a ratio of all predictions made. only useful when equal # observations for each class (which is rare)
### Cross Validation Classification Accuracy
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
scoring = 'accuracy'
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f) - Cross Validation") % (results.mean(), results.std())

### Logarithmic Loss (logloss): confidence % assigned to each prediction which weights the correct/incorrect score
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
scoring = 'neg_log_loss'
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f) - Logarithmic Loss") % (results.mean(), results.std()) # 0 represents perfect log loss

### Area Under ROC Curve (AUC): 1 is perfect, .5 is as good as random
### Sensitivity: true positive rate; Specificity: true negative rate
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
scoring = 'roc_auc'
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f) - AUC") % (results.mean(), results.std())

### Confusion Matrix: present accuracy of model with 2 or more classes
from sklearn.metrics import confusion_matrix
test_size = .33
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
model_ = LogisticRegression()
model_.fit(X_train, Y_train)
predicted = model_.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print (matrix)

### Classification Report: preicision, recall, F1-score, support
from sklearn.metrics import classification_report
report = classification_report(Y_test, predicted)
print (report)

## Regression Metrics: Mean Absolute Error, Mean Squared Error, R^2
import pandas
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
url = "https://goo.gl/sXleFv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
num_instances = len(X)
seed = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = LinearRegression()

### Mean Absolute Eror (MAE): sum of abs(predictions - actual_values)
scoring = 'neg_mean_absolute_error'
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std()) # 0 = perfect prediction

### Mean Squared Error (MSE)
scoring = 'mean_squared_error'
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())

### R^2 (R Squared): 0 means good fit, 1 means bad fit
scoring = 'r2'
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())



