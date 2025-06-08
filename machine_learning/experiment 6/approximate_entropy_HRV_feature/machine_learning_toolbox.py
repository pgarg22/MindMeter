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

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV



from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import VarianceThreshold



from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
from graphs_ml_results_toolbox import push_heatmap
import xgboost as xgb
from sklearn.svm import SVC # "Support vector classifier"  




"""
============================================================================================================
Function to see machine learning model performance
============================================================================================================
"""

def evaluate_model_performance(model,X_train,X_test,y_train,y_test,name,directory):
    model.fit(X_train, y_train)
    labels = [0, 1]
    y_pred = model.predict(X_test)
    cm= confusion_matrix(y_test,y_pred,labels=labels)
    push_heatmap(cm,name,directory)
    precision=precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1score= f1_score(y_test, y_pred)
    accuracy= accuracy_score(y_test, y_pred)
    
    print("**********************************************************************************")
    print("Model : " + name)
    print('Precision: %.3f' % precision )
    print('Recall: %.3f' % recall)
    print('F1: %.3f' % f1score)
    print('Accuracy: %.3f' % accuracy)  
    print("**********************************************************************************")
    
    return([accuracy,precision, recall, f1score])


"""
============================================================================================================
Function to do some basic filtering of features
============================================================================================================
"""
    
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
def scale_numerical_standard(data,scaler=None):
    
    flag=0
    if scaler==None :
        scaler = StandardScaler()
        flag=1
    data[data.columns] = scaler.fit_transform(data[data.columns])
    if flag==1:
        return(data,scaler)
    return (data)
    



"""
============================================================================================================
Function to optimize random forest hyperparameters
============================================================================================================
"""
def optimize_rf_hyperparameters(X_train,y_train):
    
# Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    # Set up the k-fold cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier(random_state=0)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = kfold, verbose=1, random_state=42, n_jobs = -1,scoring="accuracy")
    # Fit the random search model
    rf_random.fit(X_train, y_train)    
    return(rf_random.best_params_)
    



"""
============================================================================================================
Function to optimize Xtreme Greadient Boost hyperparameters
============================================================================================================
"""
def optimize_xgb_hyperparameters(X_train,y_train):
    
# Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 20)]
    
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(3, 110, num = 11)]
    
    learning_rate = [0.01, 0.05, 0.1,0.2,0.3]
    
    subsample = np.arange(0.5, 1.0, 0.1)
    
    colsample_bytree=  np.arange(0.4, 1.0, 0.1)
    colsample_bylevel= np.arange(0.4, 1.0, 0.1)
    
    # Create the random grid
    random_grid =  { 'max_depth': max_depth,
           'learning_rate': learning_rate,
           'subsample': subsample,
           'colsample_bytree': colsample_bytree,
           'colsample_bylevel': colsample_bylevel,
           'n_estimators': n_estimators}
    
    
    # Set up the k-fold cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    modelxgb = xgb.XGBClassifier(seed=20)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    xgb_random = RandomizedSearchCV(estimator = modelxgb, param_distributions = random_grid, n_iter = 100, cv = kfold, verbose=1, random_state=42, n_jobs = -1,scoring='accuracy')
    # Fit the random search model
    xgb_random.fit(X_train, y_train)  
    return(xgb_random.best_params_)



"""
============================================================================================================
Function to optimize Xtreme Greadient Boost hyperparameters
============================================================================================================
"""
def optimize_svc_hyperparameters(X_train,y_train,poly_flag=False):
    
    if(poly_flag):
        kernels=['rbf','poly','sigmoid']
    else:
        kernels=['rbf','sigmoid']
    
    # List of C values
    C_range = np.logspace(-10, 10, 21)
    # List of gamma values
    gamma_range = np.logspace(-10, 10, 21)
    # defining parameter range
    
    
    random_grid = {'C': C_range, 
              'gamma': gamma_range.tolist()+['scale', 'auto'],
              'kernel': kernels} 
    
    # Set up the k-fold cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    modelsvc =  SVC(random_state=0,probability=False) 
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    svc_random = RandomizedSearchCV(estimator = modelsvc, param_distributions = random_grid, n_iter = 100, cv = kfold, verbose=1, random_state=42, n_jobs = -1,scoring='accuracy')
    # Fit the random search model
    svc_random.fit(X_train, y_train)   
    return(svc_random.best_params_)



"""
============================================================================================================
Function to run random forest ML algorithm without feature selection
============================================================================================================
"""

def run_random_forest_without_fs(X_train, X_test, y_train, y_test,test_name,directory):
 
     best_rf_hyperparam= optimize_rf_hyperparameters(X_train,y_train)
     optimized_model = RandomForestClassifier(**best_rf_hyperparam)
     return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test,test_name+"_Random Forest",directory))
     

"""
============================================================================================================
Function to run XGB ML algorithm without feature selection
============================================================================================================
"""

def run_xgb_without_fs(X_train, X_test, y_train, y_test,test_name,directory):
     
     best_xgb_hyperparam= optimize_xgb_hyperparameters(X_train,y_train)
     optimized_model = xgb.XGBClassifier(**best_xgb_hyperparam)       
     return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test ,test_name+"_XGB",directory))           


"""
============================================================================================================
Function to run SVM ML algorithm  without feature selection
============================================================================================================
"""

def run_svm_without_fs(X_train, X_test, y_train, y_test,test_name,directory,poly_flag=False):
     
    best_svc_hyperparam= optimize_svc_hyperparameters(X_train,y_train,poly_flag)
    optimized_model = SVC(**best_svc_hyperparam)
    return(evaluate_model_performance(optimized_model,X_train, X_test, y_train, y_test, test_name+ "_SVM",directory))


    