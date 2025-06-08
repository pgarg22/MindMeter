#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:33:55 2022

@author: octopusphoenix
"""
import pandas as pd


df_ecg_features= pd.read_csv("ecg_features_events.csv")
df_rsp_features= pd.read_csv("rsp_features_events.csv")
new_df = pd.merge(df_ecg_features, df_rsp_features,  how='left', left_on=['Label','Participant','Condition'], right_on = ['Label','Participant','Condition'])


new_df=new_df.loc[:, ~new_df.columns.isin(['Unnamed: 0_x','Unnamed: 0_y'])]

new_df.to_csv('event_features_ecg_rsp.csv')