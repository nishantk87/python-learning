# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:13:51 2018

@author: nishantkhanna
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Import Dataset

"""

#dataset = pd.read_csv("/Users/nishantkhanna/Desktop/Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv")
dataset = pd.read_csv("C:\\Users\\nkhanna\\Desktop\\Python-Learning\\SimpleLinearRegression\\Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting data to test and training

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fiting Simple Linear Regression to training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test Set Results

y_pred = regressor.predict(X_test)

#Visualizing Training Set Results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary V/S Experience(Training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualizing Test Set Results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary V/S Experience(Test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()