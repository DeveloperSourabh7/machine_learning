# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:25:36 2019

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("student-mat.csv",sep=";")
dataset=dataset[["G1","G2","G3","studytime","failures","absences"]]
X=dataset.drop(["G3"],1).values
Y=dataset["G3"].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
