# Ch 8
# Feature Selection for Machine Learning

import pandas
import numpy as np

np.set_printoptions(precision=3)

# load data
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

# Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x, y)

print (fit.scores_) # summarize scores
features = fit.transform(x)
print (features[0:5,:])

# Feature Extraction with Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 3) # select top 3 features
fit = rfe.fit(x,y)
print ("Num Features: %d") % fit.n_features_
print ("Selected Features: %s") % fit.support_
print ("Feature Ranking: %s") % fit.ranking_

# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=3) # number of principal components
fit = pca.fit(x)
print ("Explained Variance: %s") % fit.explained_variance_ratio_
print (fit.components_)

# Feature Importance: Decision Trees
# ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
print (model.feature_importances_)
