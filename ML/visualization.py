# Chapter 6
# Understand your data with Visualization

import matplotlib.pyplot as plt
import pandas
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = pandas.read_csv(url, names=names)

# Univariate Plots: histograms, density plots, box & whisker plots

# Histograms
# See exponential dist (age, pedi, test) and gaussian dist (mass, pres, plas)
data.hist()
# plt.show()

# Density Plots: histogram but with curve not bars
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# plt.show()

# Box and Whisker Plots - Boxplots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

# Multivariate Plots: correlation matrix plot, scatter plot matrix

import numpy as np

# Correlation Matrix Plot
correlations = data.corr()

# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
# plt.show()

# Scatter Plot Matrix
from pandas.tools.plotting import scatter_matrix
scatter_matrix(data)
plt.show()
