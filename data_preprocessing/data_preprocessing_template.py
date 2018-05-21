#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:41:57 2018

@author: nishantkhanna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Import Dataset

"""

#dataset = pd.read_csv("/Users/nishantkhanna/Desktop/Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv")
dataset = pd.read_csv("C:\\Users\\nkhanna\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 1 - Data Preprocessing\\Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Missing data imputation
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features= [0])
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoder_Y = LabelEncoder()
y = labelEncoder_Y.fit_transform(y)

#Splitting data to test and training

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)