import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get("EOD/HD", authtoken="wq2asyVGTsYhXv66xkp4") # get data set

df = df[[ 'Adj_Open', 'Adj_High','Adj_Low','Adj_Close','Adj_Volume']] # adjust data set

df['HL_PCT'] = (df['Adj_High'] - df['Adj_Close']) / df['Adj_Close'] * 100.0 # Hight close PCT as new column

df['PCT_Change'] = (df['Adj_Close'] - df['Adj_Open']) / df['Adj_Open'] * 100.0 # daily change PCT as new column

df = df[[ 'Adj_Close', 'HL_PCT','PCT_Change','Adj_Volume']] # adjust data frame again for new create columns

forecast_col = 'Adj_Close' # select label column
df.fillna(-99999, inplace = True) # fill NAs

forecast_out = int( math.ceil(0.01 * len(df)) ) # length of forcast  1% of data frame for now, it can change

df['label'] = df[forecast_col].shift(-forecast_out) # shift Adj_Close -10 for now 

X = np.array(df.drop(['label'], 1)) # Features are X
X = preprocessing.scale(X) # scale X for making it faster and easier
X_lately = X[ -forecast_out : ] # forecast set

X = X[ : -forecast_out]
df.dropna(inplace = True) # clear NAs
y = np.array(df['label']) # Labels are y

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4 ) # set train data and test data

#clf = LinearRegression(n_jobs=-1) # use LinearRegression, n_jobs how many thread it will use -1 means as mush as possible

#clf.fit(X_train, y_train) # train the classifier

######## pickle classifier

#with open('Regression\linearregression.pickle', 'wb') as f:
#    pickle.dump(clf, f)
pickle_in = open('Regression\linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy =  clf.score(X_test, y_test) # test the classifier and get accuracy

forecast_set = clf.predict(X_lately) # predict as x_lately

print(forecast_set, accuracy, forecast_out) # print forecast

df['Forecast'] = np.nan # fill nan to forecast column

last_date = df.iloc[-1].name # get last date
last_unix = last_date.timestamp() # get unix of date
one_day = 86400 # set day as second
next_unix = last_unix + one_day 

for i in forecast_set: # Fill df with new forecast rows
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

print(df.tail())

