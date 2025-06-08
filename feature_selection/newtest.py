#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 18:23:42 2022

@author: octopusphoenix
"""
import numpy as np
import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from functions_toolbox import feature_selector_basic

from functions_toolbox import  transform_categorical, scale_numerical_min_max
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


df_event_features_all= pd.read_csv("event_features_ecg_rsp.csv")

X_forest=df_event_features_all.loc[:, ~df_event_features_all.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y_forest=df_event_features_all['Condition']  #

y_forest,label_encoder=transform_categorical(y_forest)
X_forest,encoder=scale_numerical_min_max(X_forest)


#X_forest=  feature_selector_basic(X_forest,y_forest,0.9)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

efs1 = EFS(clf, 
           min_features=1,
            max_features=3  ,
           scoring='accuracy',
           print_progress=True,
           cv=4,
           n_jobs=-1)

efs1 = efs1.fit(X_forest, y_forest)

print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset (indices):', efs1.best_idx_)
print('Best subset (corresponding names):', efs1.best_feature_names_)