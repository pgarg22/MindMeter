#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:50:05 2023

@author: octopusphoenix
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:56:40 2023

@author: octopusphoenix
"""

import numpy as np
import neurokit2 as nk
import pandas as pd
import os

"""
============================================================================================================
Function to clean missing values and nan's'
============================================================================================================
"""    
   
    
def clean_missing_values(df,miss_thres_perc, mean_cols=[], median_cols=[],mode_cols=[]):
    
    min_count =  int(((100-miss_thres_perc)/100)*df.shape[0] + 1)
    df= df.dropna(axis=1, 
                thresh=min_count)
    for value in mean_cols:
        if value in df.columns:
            df[value]= df[value].fillna(df[value].mean()) 	
    
    for value in median_cols:
        if value in df.columns:
            df[value]= df[value].fillna(df[value].median()) 	

    for value in mode_cols:
        if value in df.columns:
            df[value]= df[value].fillna(df[value].mode()[0]) 	

    return(df)
 

"""
============================================================================================================
Function to process ecg signal
============================================================================================================
"""    
   
       

def process_ecg_signal(data_ecg_res,condition_list,period_start,period_end,file_name,drop_missing=False):
    
    
    onset_arr= np.array([ 0,  98000,  157500, 266500,315000,473500,589000,1030000])
    duration_arr=  np.array([98000, 59500, 109000, 48500,158500,115500,441000,210000])
    
    for i in range(0,len(onset_arr)):
        onset_arr[i]= onset_arr[i]+ period_start
        duration_arr[i]= duration_arr[i]- period_start-period_end
    
    
    label_arr= np.array(['1', '2', '3', '4', '5', '6', '7', '8'], dtype='<U11')

    events= {}
    events['onset']=onset_arr
    events['duration']= duration_arr 
    events['label']= label_arr
    events['condition']= condition_list
    
    
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
    if(drop_missing):
        df_event_features.drop(nan_cols, inplace=True)
    else:
        df_event_features=   clean_missing_values(df_event_features,50, median_cols=nan_cols)

    df_event_features=df_event_features.loc[:, ~df_event_features.columns.isin(['Unnamed: 0'])]
    if not os.path.exists("ecg_processed_files"):
        os.mkdir("ecg_processed_files")
    df_event_features.to_csv("ecg_processed_files/"+ file_name+".csv")
    
    return(df_event_features)
    
    
    
    
"""
============================================================================================================
Function to process respiratory signal
============================================================================================================
"""    
       
    
def process_rsp_signal(data_ecg_res,condition_list,period_start,period_end,file_name,drop_missing=False):
    
    

    onset_arr= np.array([ 0,  98000,  157500, 266500,315000,473500,589000,1030000])
    duration_arr=  np.array([98000, 59500, 109000, 48500,158500,115500,441000,210000])
    
    for i in range(0,len(onset_arr)):
        onset_arr[i]= onset_arr[i]+ (period_start*500)
        duration_arr[i]= duration_arr[i]- ((period_start+period_end)*500)
    
    
    label_arr= np.array(['1', '2', '3', '4', '5', '6', '7', '8'], dtype='<U11')
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
    
    
    if(drop_missing):
        df_event_features_rsp.drop(nan_cols, inplace=True)
    else:
        df_event_features_rsp=   clean_missing_values(df_event_features_rsp,50, median_cols=nan_cols)

    df_event_features_rsp=df_event_features_rsp.loc[:, ~df_event_features_rsp.columns.isin(['Unnamed: 0'])]
    if not os.path.exists("rsp_processed_files"):
        os.mkdir("rsp_processed_files")
    df_event_features_rsp.to_csv("rsp_processed_files/"+ file_name+".csv")
    
    return(df_event_features_rsp)
    


"""
============================================================================================================
Function to merge results for ecg and rsp signal
============================================================================================================
"""    
   

def merge_ecg_rsp(df_ecg_features,df_rsp_features, new_name,video_label_type):

    new_df = pd.merge(df_ecg_features, df_rsp_features,  how='left', left_on=['Label','Participant','Condition'], right_on = ['Label','Participant','Condition'])
    new_df=new_df.loc[:, ~new_df.columns.isin(['Unnamed: 0_x','Unnamed: 0_y'])]
    if not os.path.exists("merged_processed_files2"):
        os.mkdir("merged_processed_files2")
    if not os.path.exists("merged_processed_files2/"+video_label_type):
        os.mkdir("merged_processed_files2/"+video_label_type)
    new_df.to_csv("merged_processed_files2/"+video_label_type+"/"+new_name+".csv")
    
    return(new_df)

def run_ecg_res_processing(data_ecg_res,condition_list,period_start,period_end,file_name,video_label_type,drop_missing=False):
    
    df1=process_ecg_signal(data_ecg_res,condition_list,period_start,period_end,"ECG_Features"+file_name,drop_missing)
    df2=process_rsp_signal(data_ecg_res,condition_list,period_start,period_end,"RSP_Features"+file_name,drop_missing)    
        
    combineddf= merge_ecg_rsp(df1,df2,"ECG_RSP_Features"+file_name,video_label_type)

    return(combineddf)

    