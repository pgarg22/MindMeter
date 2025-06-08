#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:54:45 2022

This is a script file containing all the functions for the anxiety  research

@author: octopusphoenix
"""


"""
============================================================================================================
Importing modules
============================================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV



from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import RandomizedSearchCV
import os




    
def feature_selector_basic(X_train,var_threshold):
    
    
    sel = VarianceThreshold(threshold=0.05)  # 0.01 indicates 99% of observations approximately
    sel.fit(X_train)  
    cols= X_train.columns[sel.get_support()]
    X_new= X_train[cols]
    
    cor_matrix = X_new.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > var_threshold)]
    filtered_X = X_train.drop(to_drop, axis=1)
    return(filtered_X)




"""
============================================================================================================
Function to select best features using recursive feature selection with cross validation
============================================================================================================
"""



def recursive_feature_selector_cv(X_train,y_train,classifier,crossvaltimes,scoring):
    
    rfecv = RFECV(estimator=classifier, step=1, cv=crossvaltimes,scoring=scoring)  
    rfecv = rfecv.fit(X_train, y_train)
    return( X_train.columns[rfecv.support_])

    


"""
============================================================================================================
Function to transform categorical columns into integer using label encoder
============================================================================================================
"""

def transform_categorical(df,label_encoder=None):
    flag=0
    if label_encoder==None:
        label_encoder = LabelEncoder()
        flag=1
    
    df= label_encoder.fit_transform(df)
    if flag==1:
        return(df,label_encoder)
    return (df)
        
    
    



"""
============================================================================================================
Function to scale numerical columns into integer using MinMax Scaler 

Returns scaler if none provided while calling the function
============================================================================================================
"""
def scale_numerical_min_max(data,scaler=None):
    flag=0
    if scaler==None :
        scaler = MinMaxScaler()
        flag=1
    data[data.columns] = scaler.fit_transform(data[data.columns])
    if flag==1:
        return(data,scaler)
    return (data)
    



"""
============================================================================================================
Function to scale numerical columns into integer using Standard Scaler 
============================================================================================================
"""
def scale_numerical_standard(data):
    scaler = StandardScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])
    return(data)
    
    






    