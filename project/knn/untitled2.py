# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:36:06 2019

@author: USER
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("car.data")
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]

X=pd.get_dummies(data=X,columns=['buying','maint','lug_boot','safety'])
Y=pd.get_dummies(data=Y,columns=['class'])
cols=["buying_high","maint_high","lug_boot_big","safety_high"]
X=X.drop(cols,1)
X=X.drop(["door","persons"],1)
Y=Y.drop(["acc"],1)

X=X.values
Y=Y.values

from sklearn.model_selection import train_test_split
X_train,Y_train,X_test,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=10)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)