# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:07:43 2022

@author: hp
"""
# importing the essential  labraries
import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
# load a data set
data = pd.read_csv('Salary_Data.csv')

#IV
X= data.iloc[:,0:1].values
#DV
y=data.iloc[:,-1].values


# Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size =0.2,random_state=42)
# Fitting SLR  to the training data set
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train,y_train)
# prediciting the result test
y_pred = lr.predict(X_train)
#  Visualising the training the data set results
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,lr.predict(X_train),color='blue')
plt.title('Salary VS Exp.')
plt.xlabel('YE')
plt.ylabel('Salary')
plt.show()

#  Visualising the testing the data set results
plt.scatter(X_test,y_test, color='red')
plt.plot(X_test,lr.predict(X_test),color='blue')
plt.title('Salary VS Exp.')
plt.xlabel('YE')
plt.ylabel('Salary')
plt.show()

