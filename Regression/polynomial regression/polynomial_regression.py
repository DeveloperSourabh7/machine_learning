# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:16:43 2019

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1].values
Y=dataset.iloc[:,2].values

x=np.reshape(X,(-1,1))
y=np.reshape(Y,(-1,1))

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=6)
X_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(X_poly),color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising the polynomial Regression results (for higher and smooth curve)
X_grid=np.arrange(min(x),max(x),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(x,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting new results with linear regression
lin_reg.predict(np.reshape(12,(-1,1)))

lin_reg2.predict(poly_reg.fit_transform(np.reshape(12,(-1,1))))