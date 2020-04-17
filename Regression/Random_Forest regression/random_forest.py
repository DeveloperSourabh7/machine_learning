# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:59:20 2019

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pdf

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,Y)

y_pred=regressor.predict(np.reshape(2.7,(-1,1)))

X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()