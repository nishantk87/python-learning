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