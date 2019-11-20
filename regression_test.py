# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:16:31 2019

@author: 19233292
"""
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR
from sklearn import linear_model
import matplotlib.pyplot as plt 
lines =[]
try:
    file = open("steel.txt")
    line = file.readline()
    while line : 
        lines.append(line.split())
        line = file.readline()
except :
    print("Unable to proceed without hazelnuts.txt. Please sure that it is in cwd.")
finally :
    file.close()
    
col = ["Normalising Temperature","Tempering Temperature", "Sample", "Percent Silicon", "Percent Chromium", "Manufacture Year", "Percent Copper",
       "Percent Nickel", "Percent Sulphur", "Percent Carbon", "Percent Manganese", "Tensile Strength"]

data = np.array(lines)
dataset = pd.DataFrame(data, columns = col)
#print(dataset.head)

dataset = dataset.sample(frac=1).reset_index(drop=True)
train, test = train_test_split(dataset, test_size=0.33)
features_data_train = train.iloc[:,:-1]
class_data_train = train['Tensile Strength']
features_data_test = test.iloc[:,:-1]
class_data_test = test['Tensile Strength']

regr = LinearRegression() 
  
regr.fit(features_data_train, class_data_train) 
print("Linear Regression : ",regr.score(features_data_test, class_data_test)) 

clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(features_data_train, class_data_train) 

print("Support Vector Regression : ",clf.score(features_data_test, class_data_test)) 

reg = linear_model.Ridge(alpha=.5)
reg.fit(features_data_train, class_data_train) 
print("Ridge Regression : ",reg.score(features_data_test, class_data_test)) 

reg1 = linear_model.Lasso(alpha=0.1)
reg1.fit(features_data_train, class_data_train) 
print("Lasso Regression : ",reg1.score(features_data_test, class_data_test)) 

