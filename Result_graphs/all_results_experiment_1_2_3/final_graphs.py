#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 01:31:20 2023

@author: octopusphoenix
"""
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

def round_up_list(my_list):
    rounded_list= [ '%.1f' % elem for elem in my_list ]
    return(rounded_list)

def percentage_list(my_list):
    percent_list= [ 100*elem for elem in my_list]
    return(percent_list)


from plotly.subplots import make_subplots


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

all_accuracies_rf=[]
all_accuracies_svm=[]
all_accuracies_xgb=[]
for experiment in experiments:
    for num_feats in num_feats_list:
        for tr_period in trans_names:
            for selector in selectors_list:
                for result_type in results:
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
                    all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                    all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                    all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())





all_accuracies=all_accuracies_rf+all_accuracies_xgb+all_accuracies_svm


fig = px.histogram(all_accuracies,nbins=20, marginal="rug")
fig.update_layout(showlegend=False, xaxis_title="Model Accuracy", yaxis_title="Count")
fig.show()
if not os.path.exists("Model_Trends"):
     os.mkdir("Model_Trends")
fig.write_image("Model_Trends/anxiety_possible.png")

fig = go.Figure(data=[go.Histogram(x=all_accuracies, nbinsx=40,cumulative_enabled=True)])
fig.update_layout(showlegend=False, xaxis_title="Model Accuracy", yaxis_title="Count")
fig.show()
if not os.path.exists("Model_Trends"):
     os.mkdir("Model_Trends")
fig.write_image("Model_Trends/anxiety_possible4.png")



all_accuracies_rf=[]
all_accuracies_svm=[]
all_accuracies_xgb=[]
for experiment in experiments:
    for num_feats in num_feats_list:
        for tr_period in trans_names:
            for selector in selectors_list:
                for result_type in results:
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
                    
                    mean= result_csv.mean()
                    all_accuracies_rf.append( mean["rf_accuracy"])
                    all_accuracies_xgb.append( mean["xgb_accuracy"])
                    all_accuracies_svm.append( mean["svm_accuracy"])





all_accuracies1=all_accuracies_rf+all_accuracies_xgb+all_accuracies_svm

all_accuracies1=percentage_list(all_accuracies)


fig = px.histogram(all_accuracies1, nbins= 20, labels={'x':'Model Accuracy', 'y':'Count'}, marginal="rug")
fig.update_layout(showlegend=False, xaxis_title="Model Accuracy", yaxis_title="Count")
fig.show()
if not os.path.exists("Model_Trends"):
     os.mkdir("Model_Trends")
fig.write_image("Model_Trends/anxiety_possible2.png")



fig = go.Figure(data=[go.Histogram(x=all_accuracies1,cumulative_enabled=True)])
fig.update_layout(showlegend=False, xaxis_title="Model Accuracy", yaxis_title="Count")
fig.show()
if not os.path.exists("Model_Trends"):
     os.mkdir("Model_Trends")
fig.write_image("Model_Trends/anxiety_possible3.png")



layout = go.Layout(
    autosize=False,
    width=2000,
    height=1000,
    )

fig = make_subplots(rows=1, cols=2,horizontal_spacing = 0.05,
                vertical_spacing= 0.02,shared_xaxes=True,subplot_titles=["Distribution of Model Accuracies", "Cumalative Distribution of Model Accuracies"])


fig.add_trace(go.Histogram(x=all_accuracies, nbinsx=20), row=1, col=1)
fig.add_trace(go.Histogram(x=all_accuracies,nbinsx=20,cumulative_enabled=True), row=1, col=2)
fig.update_layout(showlegend=False, xaxis_title="Model Accuracy", yaxis_title="Count")
fig.update_layout(height=900, width=1200,
               yaxis_zeroline=False, xaxis_zeroline=False, font=dict(
                 size=20,
             )
               )
fig.update_layout( legend=dict(font=dict(size=16)))
fig.update_annotations(font=dict(size=20))
fig.show()
if not os.path.exists("Model_Trends"):
     os.mkdir("Model_Trends")
fig.write_image("Model_Trends/anxiety_possible5.png")

print(np.mean(all_accuracies))
print(np.std(all_accuracies))


