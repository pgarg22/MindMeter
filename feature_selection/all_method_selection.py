#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:24:55 2023

@author: octopusphoenix
"""

from feature_selection_toolbox import cor_selector,rfe_logistic,embedded_lr_feature_selector
from feature_selection_toolbox import embedded_rf_feature_selector,chi_selector
from feature_selection_toolbox import Extra_tree_selector,select_k_best_score_selector
from feature_selection_toolbox import forward_feature_selector,forward_feature_selector_floating
from feature_selection_toolbox import backward_feature_selector,backward_feature_selector_floating
from feature_selection_toolbox import mutual_info_selector,mutual_info_selector2
import pandas as pd
import numpy as np
from functions_toolbox import transform_categorical

df_event_features= pd.read_csv("event_features_ecg_rsp.csv")

X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y=df_event_features['Condition']  #
num_feats=10
X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
X_train= feature_selector_basic(X_train,0.9)
X_train=X
y_train=y
y_train, encoder= transform_categorical(y_train)

cor_support, cor_feature = cor_selector(X_train, y_train,num_feats)
rfe_support, rfe_logistic_features= rfe_logistic(X_train, y_train,num_feats)
embeded_lr_support,embeded_lr_feature= embedded_lr_feature_selector(X_train, y_train,num_feats)
embeded_rf_support,embeded_rf_feature= embedded_rf_feature_selector(X_train, y_train,num_feats)
chi_support, chi_feature=  chi_selector(X_train, y_train,num_feats)


ets_support, ets_feature = Extra_tree_selector(X_train, y_train,num_feats)
skbs_support, skbs_features= select_k_best_score_selector(X_train, y_train,num_feats)
ffs_support,ffs_feature= forward_feature_selector(X_train, y_train,num_feats)
ffsf_support,ffsf_feature= forward_feature_selector_floating(X_train, y_train,num_feats)
bfs_support, bfs_feature=  backward_feature_selector(X_train, y_train,num_feats)



bfsf_support, bfsf_feature = backward_feature_selector_floating(X_train, y_train,num_feats)
mis_support, mis_features= mutual_info_selector(X_train, y_train,num_feats)
mis2_support,mis2_feature= mutual_info_selector2(X_train, y_train,num_feats)



feature_name=X_train.columns



# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 
                                     'Chi-2':chi_support, 'RFE':rfe_support, 
                                     'Logistics':embeded_lr_support,'Random Forest':embeded_rf_support
                                    , 'Extra Tree':ets_support, 'K-Best':skbs_support, 
                                    'Forward':ffs_support,'Forward floating':ffsf_support,
                                    'Backward':bfs_support, 'Backward floating':bfsf_support, 
                                    'Mutual Info':mis_support,'Mutual Info2':mis2_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)



feature_selection_df.to_csv("all_feature_selection_methods_results.csv")
sequential_feature_selection_df= pd.DataFrame({'Feature':feature_name, 
                                    'Forward':ffs_support,'Forward floating':ffsf_support,
                                    'Backward':bfs_support, 'Backward floating':bfsf_support, 
                                    })
sequential_feature_selection_df['Total'] = np.sum(sequential_feature_selection_df, axis=1)
# display the top 100
sequential_feature_selection_df = sequential_feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
sequential_feature_selection_df.index = range(1, len(sequential_feature_selection_df)+1)
sequential_feature_selection_df.head(num_feats)

sequential_feature_selection_df.to_csv("sequential_feature_selection_results.csv")
