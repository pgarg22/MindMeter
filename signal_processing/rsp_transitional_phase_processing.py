#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:10:12 2023

@author: octopusphoenix
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:23:14 2022

@author: octopusphoenix
"""


from Load_ecg_res_data import *
from signal_processing_toolbox import clean_missing_values
import numpy as np
import neurokit2 as nk
import pandas as pd



#############################################################################################
#Setting up event info



onset_arr= np.array([ 7500,  105500,  165500, 274000,322500,481000,596500,1037500])
duration_arr=  np.array([83000, 44500, 94000, 48500,33500,100500,426000,195000])
label_arr= np.array(['1', '2', '3', '4', '5', '6', '7', '8'], dtype='<U11')
condition_list = ["Anxious", "Normal", "Anxious", "Normal","Anxious", "Normal","Anxious", "Normal"]

events= {}
events['onset']=onset_arr
events['duration']= duration_arr 
events['label']= label_arr
events['condition']= condition_list



event_related_features_rsp=[]

for k in range(0,len(data_ecg_res)):
    #Clean data    
    data_clean, info = nk.bio_process(rsp=data_ecg_res[k][:,1], sampling_rate=500)
    
    # Build epochs
    epochs = nk.epochs_create(data_clean, events, sampling_rate=500)
    
    #analyze
    df = nk.bio_analyze(epochs, sampling_rate=500)
    
    df["Condition"]=condition_list
    df["Participant"]=[k+1]*8
    event_related_features_rsp.append(df)
    
    
    
df_event_features_rsp = pd.concat(event_related_features_rsp)  
df_event_features_rsp.replace([np.inf, -np.inf], np.nan, inplace=True)
nan_cols= df_event_features_rsp.columns[df_event_features_rsp.isnull().any()]

df_event_features_rsp=   clean_missing_values(df_event_features_rsp,50, median_cols=nan_cols)

df_event_features_rsp=df_event_features_rsp.loc[:, ~df_event_features_rsp.columns.isin(['Unnamed: 0'])]
df_event_features_rsp.to_csv('rsp_features_events_transitional_phase.csv')



