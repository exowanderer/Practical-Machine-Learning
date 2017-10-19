from __future__ import print_function
import numpy as np
import pandas as pd
import quandl, math

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from time import time

start = time()
'''
`preprocessing` will be used to 'scale' the labels from 0 to 1
'''

df0  = quandl.get('WIKI/GOOG')

print("Downloading the DATA took {0:.2e} seconds".format(time() - start))
start = time()
df  = pd.DataFrame.copy(df0[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']])

df['HL_PCT']    = (df['Adj. High'] - df['Adj. Low'])   / df['Adj. Close'] * 100.0
df['PCT_change']= (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']  * 100.0

df  = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col    = 'Adj. Close'
df.fillna(-99999, inplace=True) # fill "Not Applicable" (usually NaNs) with "-99999" in place

percentchange   = 0.01 # predict 1% forward from current data set
forecast_out    = int(math.ceil(percentchange*len(df)))

df['label']     = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

''' Define X and Y '''
# df.dropna(inplace=True)
X   = np.array(df.drop(['label'], 1))   # features are everything except for the `labels`
y   = np.array(df['label'])             # labels are the `labels`

''' Preprocessing can add to processing time 
        because you have to rescale testing data and prediction data 
        alongside training data for a proper scaling

    With HF Trading, you would definitely skip this step
'''
start = time()

X   = preprocessing.scale(X)
# X   = X[:-forecast_out+1]

X_train, X_test, y_train, y_test    = cross_validation.train_test_split(X, y, test_size=0.2)

print("Setting up the DATA took {0:.2e} seconds".format(time() - start))

''' LinearRegression '''
start = time()

clf = LinearRegression() # default
clf.fit(X_train, y_train)

accuracy    = clf.score(X_test,y_test)*100.0

print("Linear Regression took {0:.2e} seconds, with an accuracy of {1:.2f}%".format(time() - start, accuracy))
start = time()

''' LinearRegression Multiprocessing'''
start = time()

clf = LinearRegression(n_jobs=-1) # multiprocessing
clf.fit(X_train, y_train)

accuracy    = clf.score(X_test,y_test)*100.0

print("Linear Regression took {0:.2e} seconds, with an accuracy of {1:.2f}%".format(time() - start, accuracy))
start = time()

''' SVM Regression ''' 

clf = svm.SVR()
clf.fit(X_train, y_train)

accuracy    = clf.score(X_test,y_test)*100.0
# print("The accuracy was {0:.2e}".format(accuracy))

print("SVM Linear Kernel took {0:.2e} seconds, with an accuracy of {1:.2f}%".format(time() - start, accuracy))
start = time()

''' SVM Regression Polynomial''' 

clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)

accuracy    = clf.score(X_test,y_test)*100.0
# print("The accuracy was {0:.2e}".format(accuracy))

print("SVM Poly Kernel took {0:.2e} seconds, with an accuracy of {1:.2f}%".format(time() - start, accuracy))
start = time()

''' SVM Regression RBF (Gaussian)''' 

clf = svm.SVR(kernel='rbf')
clf.fit(X_train, y_train)

accuracy    = clf.score(X_test,y_test)*100.0
# print("The accuracy was {0:.2e}".format(accuracy))

print("SVM RBF Kernel took {0:.2e} seconds, with an accuracy of {1:.2f}%".format(time() - start, accuracy))
