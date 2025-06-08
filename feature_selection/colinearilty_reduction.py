#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:21:35 2023

@author: octopusphoenix
"""



# Import required libraries
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


    
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

import numpy as np


df_event_features= pd.read_csv("event_features_ecg_rsp.csv")

X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y=df_event_features['Condition']  #

sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)
vif_info.reset_index(inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)



vif_info.reset_index(inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)




vif_info.reset_index(inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)


vif_info.reset_index(inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)



vif_info.reset_index(inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)



vif_info.reset_index(inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)



drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:9].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)


drop_columns= vif_info.reset_index()['Column'][0:4].tolist()


X = X.drop(drop_columns, axis=1)

vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_info['Column'] = X.columns
vif_info.sort_values('VIF', ascending=False, inplace=True)


