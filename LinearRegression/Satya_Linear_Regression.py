# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:49:50 2020

@author: U58722
"""

#simple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.333, random_state = 0)

#fitting Simple Linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the Test set Results
y_pred = regressor.predict(X_test)

#visualization of training set results
plt.scatter(X_train , y_train , color = 'red')
plt.plot(X_train, regressor.predict(X_train) , color = 'blue')
plt.title('Salary vs Experince(TrainingSet)')
plt.xlabel('years of experince')
plt.ylabel('Salary')
plt.show()

