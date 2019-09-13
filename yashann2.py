# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 07:09:36 2019

@author: yash teotia
"""
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing the dataset 
 dataset= pd.read_csv('Churn_Modelling.csv')
 X = dataset.iloc[: ,3:13].values
 y = dataset.iloc[:,13].values
# preventing variable trap 
 from sklearn.preprocessing import LabelEncoder  
 labelencoder = LabelEncoder()
  X[:,1] = labelencoder.fit_transform(X[:,1])
  labelencoder2 =LabelEncoder()
  X[:,2] = labelencoder2.fit_transform(X[:,2])
   from sklearn.preprocessing import OneHotEncoder 
  onehotencoder = OneHotEncoder(categorical_features=[1])
 X = onehotencoder.fit_transform(X).toarray()
 X = X[:,1:]

# Train test split 
from sklearn.model_selection import train_test_split
X_train ,X_test,y_train , y_test = train_test_split(X,y,test_size = 0.2 , random_state= 0)
# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

import keras
from keras.layers import Dense
from keras.models import Sequential


classifier = Sequential()

classifier.add(Dense(units = 6 , init = 'uniform' , activation = 'relu' , input_dim = 11))
classifier.add(Dense(units = 6 , init = 'uniform' , activation = 'relu' ))
classifier.add(Dense(units = 6 , init = 'uniform' , activation = 'relu' ))
classifier.add(Dense(units = 6 , init = 'uniform' , activation = 'relu' ))
classifier.add(Dense(units = 1 , init = 'uniform' , activation = 'sigmoid'))

classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
 
classifier.fit(X_train , y_train , batch_size = 10 , nb_epoch=100 )

y_pred =classifier.predict(X_test)
y_pred = (y_pred>0.5)
# confusion matrix 
from sklearn.metrics import confusion_matrix
confusionmatrix_ann = confusion_matrix(y_test,y_pred)

# ACCURACY OF 84.25%






