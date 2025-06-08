#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:17:55 2023

@author: octopusphoenix
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:01:19 2023

@author: octopusphoenix
"""


from data_split_toolbox  import run_multiple_tests_without_fs_trans
import pandas as pd
from feature_selection_toolbox import run_all_selectors
from sklearn.model_selection import train_test_split
from machine_learning_toolbox import feature_selector_basic, transform_categorical, scale_numerical_standard
import os
import warnings
warnings.filterwarnings("ignore")



names=["trans0", "trans10","trans20","trans30","trans10_onlybegin","trans20_onlybegin",
       "trans30_onlybegin","trans40_onlybegin"]





for c in  names:
        
        df_event_features_trans= pd.read_csv("processed_files/RSP_Features"+c+".csv")
        X_trans=df_event_features_trans.loc[:, ~df_event_features_trans.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
        y_trans=df_event_features_trans['Condition']  #
        
        
        
        X_trans, train_scaler_trans=scale_numerical_standard(X_trans)
        y_trans,encoder_trans= transform_categorical(y_trans)
        

        features_list_trans=[['RRV_ApEn', 'Condition','Label','Unnamed: 0','Participant'], ['RRV_ApEn','RRV_LFn', 'Condition','Label','Unnamed: 0','Participant'], ['RRV_ApEn','RRV_LFn', 'RRV_HFn', 'Condition','Label','Unnamed: 0','Participant']]
        selector_names_trans=['RRV_ApEn', 'RRV_ApEn_LFn', 'RRV_ApEn_LFn_HFn']
        
             
        for n,name in enumerate(selector_names_trans):
            selector_names_trans[n]=selector_names_trans[n]+"_"+c
        
        run_multiple_tests_without_fs_trans(df_event_features_trans,
                                      features_list_trans,selector_names_trans)
        
    