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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)