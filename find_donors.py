# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
from IPython import get_ipython
from sklearn.cross_validation import train_test_split


# Import supplementary visualization code visuals.py
# import visuals as vs
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
import random
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.base import clone
# Import supplementary visualization code visuals.py
# import vpython as vs

# Pretty display for notebooks
# %matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
# plt.show()


n_records = data['age'].count()
df = data['income']

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = df[df=='>50'].count()

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = df[df=='<=50'].count()

# TODO: Percentage of individuals whose income is more than $50,000
x = n_greater_50k
y = n_records

greater_percent = float((float(x) / float(y))*100)

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)
