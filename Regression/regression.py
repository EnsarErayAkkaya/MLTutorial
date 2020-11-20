import pandas as pd
import quandl, math
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression


df = quandl.get("EOD/HD", authtoken="wq2asyVGTsYhXv66xkp4") # get data set

df = df[[ 'Adj_Open', 'Adj_High','Adj_Low','Adj_Close','Adj_Volume']] # adjust data set

df['HL_PCT'] = (df['Adj_High'] - df['Adj_Close']) / df['Adj_Close'] * 100.0 # Hight close PCT as new column

df['PCT_Change'] = (df['Adj_Close'] - df['Adj_Open']) / df['Adj_Open'] * 100.0 # daily change PCT as new column

df = df[[ 'Adj_Close', 'HL_PCT','PCT_Change','Adj_Volume']] # adjust data frame again for new create columns

forecast_col = 'Adj_Close' # select label column
df.fillna(-99999, inplace = True) # remove NAs

forecast_out = int( math.ceil(0.01 * len(df)) ) # length of forcast  1% of data frame for now, it can change

df['label'] = df[forecast_col].shift(-forecast_out) # shift Adj_Close -10 for now 
df.dropna(inplace = True)

# Features are X
X = np.array(df.drop(['label'], 1))
# Labels are y
y = np.array(df['label'])   

X = preprocessing.scale(X) # scale X for making it faster and easier

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4 ) # set train data and test data

clf = LinearRegression() # use LinearRegression

clf.fit(X_train, y_train) # train the classifier

accuracy =  clf.score(X_test, y_test) # test the classifier and get accuracy


print(accuracy)
