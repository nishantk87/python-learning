# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:27:28 2018

@author: nishantkhanna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Import Dataset

"""

#dataset = pd.read_csv("/Users/nishantkhanna/Desktop/Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv")
dataset = pd.read_csv("C:\\Users\\nkhanna\\Desktop\\Python-Learning\\MultipleLinearRegression\\50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
oneHotEncoder = OneHotEncoder(categorical_features= [3])
X = oneHotEncoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X = X[:, 1:]

#Splitting data to test and training

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression Model to training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test Set Results

y_pred = regressor.predict(X_test)

#building optimal model using Backward elimination
import statsmodels.formula.api as sm

X = np.append(np.ones((50,1)).astype(int),X, axis = 1 )