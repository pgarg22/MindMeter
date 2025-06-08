#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:19:51 2023

@author: octopusphoenix
"""

from experiment_plots import *


num_feats_list=[5,10]

trans_names=["trans0", "trans10","trans20","trans30"
         ,"trans10_onlybegin","trans20_onlybegin",
        "trans30_onlybegin","trans40_onlybegin"]

selectors_list=['Pearson Coefficient Feature Selector',
                'Logistic Regression Feature Selector',
                'Random Forest Feature Selector',
                'Extra Tree Feature Selector',
                'Chi-2 Feature Selector',
                'Sequential Forward Feature Selector',
                'Sequential Backward Feature Selector',
                'Mutual Info Feature Selector',
                'Factor Analysis Selector',
                'PCA selector',
                'Best All Selectors']


metric_list=['accuracy','recall','precision','f1score']
experiments=["experiment1","experiment2","experiment3"]
results=["Participant","Video","random"]
label= ["ECG", "RSP","ECG_RSP"]

distribution_plot_all(experiments,num_feats_list,trans_names,selectors_list,results)
for i, experiment in enumerate(experiments):
    print(experiment)
    print("\n")
    
    distribution_plot_split(experiment,num_feats_list,trans_names,selectors_list,results,label[i])
    
    
    
    
    distribution_plot(experiment,num_feats_list,trans_names,selectors_list,results,label[i])
    
    
    plot_best_data_split(experiment,num_feats_list,trans_names,selectors_list,results,label[i])
    features_used(experiment,num_feats_list,trans_names,selectors_list,results,label[i])
    plot_best(experiment,num_feats_list,trans_names,selectors_list,results,label[i])
    best_accuracy_overall(experiment,num_feats_list,trans_names,selectors_list,results)
    best_trans_overall(experiment,num_feats_list,trans_names,selectors_list,results)
    best_selector_overall(experiment,num_feats_list,trans_names,selectors_list,results)
    best_selector_overall_withnumber(experiment,num_feats_list,trans_names,selectors_list,results)
    plot_std_bar_graph_subplots(experiment,num_feats_list,trans_names,selectors_list,results)
    
