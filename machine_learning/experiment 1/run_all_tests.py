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


num_feats_list=[5,10]

names=["trans0", "trans10","trans20","trans30","trans10_onlybegin","trans20_onlybegin",
       "trans30_onlybegin","trans40_onlybegin"]





# df_metadata= pd.read_csv("processed_files/Metadata.csv")
# cols=["Participant ID","Beck Anxiety","Hamilton Anxiety"]
# df_metadata=df_metadata[cols]



for num_feats in  num_feats_list:
    for c in  names:
        
        df_event_features_trans= pd.read_csv("processed_files/ECG_Features"+c+".csv")
        X_trans=df_event_features_trans.loc[:, ~df_event_features_trans.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
        y_trans=df_event_features_trans['Condition']  #
        
        
        
        X_trans, train_scaler_trans=scale_numerical_standard(X_trans)
        y_trans,encoder_trans= transform_categorical(y_trans)
        
        
        # X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_trans, y_trans, test_size=0.1, stratify=y_trans, random_state=0)
        
        # y_train_trans, encoder_trans= transform_categorical(y_train_trans)
        # X_train_trans, train_scaler_trans=scale_numerical_standard(X_train_trans)
        # X_train_trans= feature_selector_basic(X_train_trans,0.9)
        
        df_feature_selector_trans, features_list_trans, selector_names_trans= run_all_selectors(X_trans,y_trans,num_feats)
        if not os.path.exists("feature_lists"):
            os.mkdir("feature_lists")
        df_feature_selector_trans.to_csv("feature_lists/"+ c+"features" +str(num_feats)+".csv")
        best_features_trans= df_feature_selector_trans.head(num_feats)["Feature"].to_list()                                  
        features_list_trans.append(best_features_trans)
        selector_names_trans.append("Best All Selectors")
        for feat in features_list_trans:
            feat.append('Condition')
            feat.append('Label')
            feat.append('Unnamed: 0')
            feat.append('Participant')
            
        for n,name in enumerate(selector_names_trans):
            selector_names_trans[n]=selector_names_trans[n]+"_"+c+"_"+"features"+str(num_feats)
        run_multiple_tests_without_fs_trans(df_event_features_trans,
                                      features_list_trans,selector_names_trans)
        
    
