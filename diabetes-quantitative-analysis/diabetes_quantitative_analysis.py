# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:33:41 2019

@author: 18193
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


from sklearn.datasets import load_diabetes


dataBunch = load_diabetes()

dataSet = dataBunch.data

dataSet = pd.DataFrame(data = dataSet, columns = dataBunch.feature_names)

y = dataBunch.target

#Spltting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataSet, y, test_size=0.2, random_state=0)

import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((353, 1)).astype(int), values = X_train, axis = 1) #add a column to the matrix

#first iteration
X_opt = X_train[:, [0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

#second iteration
X_opt = X_train[:, [0,1,2,3,4,5,6,8,9,10]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

#third iteration
X_opt = X_train[:, [0,2,3,4,5,6,8,9,10]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

#fourth iteration
X_opt = X_train[:, [0,2,3,4,5,6,8,9]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

#fifth iteration
X_opt = X_train[:, [0,2,3,4,5,8,9]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_test = np.append(arr = np.ones((89, 1)).astype(int), values = X_test, axis = 1) #add a column to the matrix
y_pred = regressor_OLS.predict(X_test[:, [0,2,3,4,5,8,9]])

y_pred = pd.DataFrame(data = y_pred)

# Visualising the Linear Regression results
plt.title('Quantitative Diabetes Progression')
plt.scatter(y_pred, y_test, color = 'red')
plt.xlabel('actual diabetes progression')
plt.ylabel('predicted diabetes progression')
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()