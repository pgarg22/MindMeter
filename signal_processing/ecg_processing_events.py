#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:22:51 2022

@author: octopusphoenix
"""

from Load_ecg_res_data import *
from signal_processing_toolbox import clean_missing_values
import numpy as np
import neurokit2 as nk
import pandas as pd



#############################################################################################
#Setting up event info

onset_arr= np.array([ 0,  98000,  157500, 266500,315000,473500,589000,1030000])
duration_arr=  np.array([98000, 59500, 109000, 48500,158500,115500,441000,210000])
label_arr= np.array(['1', '2', '3', '4', '5', '6', '7', '8'], dtype='<U11')
condition_list = ["Anxious", "Normal", "Anxious", "Normal","Anxious", "Normal","Anxious", "Normal"]

events= {}
events['onset']=onset_arr
events['duration']= duration_arr 
events['label']= label_arr
events['condition']= condition_list





#############################################################################################


event_related_features_ecg=[]

for k in range(0,len(data_ecg_res)):
    #Clean data    
    data_clean, info = nk.bio_process(ecg=data_ecg_res[k][:,0], sampling_rate=500)
    
    # Build epochs
    epochs = nk.epochs_create(data_clean, events, sampling_rate=500)
    
    #analyze
    df = nk.bio_analyze(epochs, sampling_rate=500)
    
    df["Condition"]=condition_list
    df["Participant"]=[k+1]*8
    event_related_features_ecg.append(df)
    
    
    
df_event_features = pd.concat(event_related_features_ecg)  

nan_cols= df_event_features.columns[df_event_features.isnull().any()]

df_event_features=   clean_missing_values(df_event_features,50, median_cols=nan_cols)

df_event_features=df_event_features.loc[:, ~df_event_features.columns.isin(['Unnamed: 0'])]

df_event_features.to_csv('ecg_features_events.csv')
    