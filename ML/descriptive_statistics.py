# Chapter 5
# Understand your data with Descriptive Statistics
import pandas
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

# Peek at Data
peek = data.head(20)
# print peek

# Dimensions of Data
# too many rows and algorithms take too long to train, too few and you might not have enough data
# too many features and some algos can have poor performance
shape = data.shape
# print (shape)

# Data type for each Attribute
# strings may need to be converted to numbers to represent categorical or ordinal values
types = data.dtypes
# print (types)

# Descriptive Statistics
# count, mean, standard deviation, minimum value, 25th %, 50th %, 75th %, max val
pandas.set_option('display.width', 100)
pandas.set_option('precision', 3)
description = data.describe()
# print (description)

# Class Distributions (Classification Problems Only)
# highly imbalanced problems (a lot more observations for one class than another) are common
# and may need special handling in the data preperation stage
class_counts = data.groupby('class').size()
# print (class_counts)

# Correlations Between Attributes
# relationship b/w 2 variables and how they may/not change together
# Pearson's Correlation Coefficient: most common way to calculate correlation (assumes normal distribution)
# -1 or 1 shows full negative/positve correlation
# 0 shows no correlation
# linear/logistic regressino suffer poor performance if there are high correlation in dataset
correlations = data.corr(method='pearson')
# print (correlations)

# Skew of Univariate Distributions 
# gaussian (normal/bell) that is shifted or squashed in one direction
# most ML algos assume gaussian distribution
# knowing an attribute has a skew may allow you to correct it to improve accuracy
# 0 = no skew
skew = data.skew()
print (skew)
