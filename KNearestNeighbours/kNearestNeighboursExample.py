import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, neighbors
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) # gets current directory

df = pd.read_csv(os.path.join(__location__, 'breast-cancer-wisconsin.txt')) # get file in directory

df.replace('?', -99999, inplace = True) # int his dataset all empty values marked by '?' we cahnged to -99999 for making them unused
df.drop(['id'], 1, inplace = True) # id is irrevelant for this example so we are droping it

X = np.array( df.drop(['class'], 1) ) # X is our attributes and everything except class is attribute
y = np.array( df['class'] ) # y is label and it is class

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) 

clf = neighbors.KNeighborsClassifier() # use KNearestNeighbours
clf.fit(X_train, y_train)# train it

accuracy = clf.score(X_test, y_test) # test it
print(accuracy)

example_measures = np.array([[4,3,3,2,2,2,1,3,1], [3,3,4,3,3,4,3,4,3]]) # create example data 
example_measures = example_measures.reshape(len(example_measures), -1) # reshape example data

prediction = clf.predict(example_measures) # predict is it benign or malignant
print(prediction)
