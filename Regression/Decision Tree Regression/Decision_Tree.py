# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:35:47 2019

@author: USER
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#Fitting Decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

#Predicting a new result
y_pred=regressor.predict(np.reshape(2.7,(-1,1)))

#Visualising the Decision Tree Regression results(higher resolution)
#X_grid=np.arange(min(X),max(X),0.01)
#X_grid=X_grid.resape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()