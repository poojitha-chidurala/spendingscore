# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
dataset=pd.read_csv("Mall_Customers.csv")
dataset

dataset.isnull().any()
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=dataset.iloc[:,4:].values
y
x=dataset.iloc[:,:4].values
x
x[:,1]=lb.fit_transform(x[:,1])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
mr=LinearRegression()
mr.fit(x_train,y_train)# training process
y_predict_mr=mr.predict(x_test)
y_predict_mr
mr.predict([[1, 1, 19, 15]])
mr.score(x_train,y_train)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict_mr)

from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)
y_predict_dt =regressor.predict(x_test)
y_predict_dt
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
regressor.predict([[1, 1, 19, 15]])
from sklearn.metrics import r2_score
r2_score(y_test,y_predict_dt)

