#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:08:46 2023

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
    rounded_list= [ '%.2f' % elem for elem in my_list ]
    return(rounded_list)

def percentage_list(my_list):
    percent_list= [ 100*elem for elem in my_list]
    return(percent_list)


marker_colors=["#bdcf32","#edbf33","#b33dc6"]


from plotly.subplots import make_subplots



def distribution_plot_split(experiment,num_feats_list,trans_names,selectors_list,results,label):
    
    
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    
    for result_type in results:
        all_accuracies=[]
        for num_feats in num_feats_list:
            for tr_period in trans_names:
                    for selector in selectors_list:
                        all_accuracies_rf=[]
                        all_accuracies_svm=[]
                        all_accuracies_xgb=[]
                    
                        test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                        result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                        result_csv.drop("Unnamed: 0",axis=1,inplace=True)
            
                        all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                        all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                        all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())

                        all_accuracies.append(np.mean(all_accuracies_rf))
                        all_accuracies.append(np.mean(all_accuracies_xgb))
                        all_accuracies.append(np.mean(all_accuracies_svm))


        fig = make_subplots(rows=1, cols=2,horizontal_spacing = 0.05,
                    vertical_spacing= 0.02,shared_xaxes=True,subplot_titles=["Distribution of Model Accuracies", "Cumulative Distribution of Model Accuracies"])

        
        
        if(result_type =="random"):
            result_type="Random"
        fig.update_layout(layout)
        fig.add_trace(go.Histogram(x=all_accuracies, nbinsx=10), row=1, col=1)
        fig.add_trace(go.Histogram(x=all_accuracies,nbinsx=10,histnorm='probability',cumulative_enabled=True), row=1, col=2)
        fig.update_layout(showlegend=False, xaxis_title="Model Accuracy", yaxis_title="Count",
                          title_text=result_type+ "_split results for models using " + label + " signal")
        fig.update_layout(height=700, width=1200,
                       yaxis_zeroline=False, xaxis_zeroline=False, font=dict(color='white',
                         size=18,
                     )
                       )
        fig.update_layout( legend=dict(font=dict(size=16)))
        fig.update_annotations(font=dict(size=18))
        fig.show()
        if not os.path.exists("Model_Trends4"):
             os.mkdir("Model_Trends4")
        fig.write_image("Model_Trends4/anxiety_dist_"+experiment+"_"+result_type+".png")
        
        
        print("result_type:" + result_type)
        print(np.mean(all_accuracies))
        print(np.std(all_accuracies))
        
        
        
        count=0
        for c in all_accuracies:
            if(c>=75):
                print(c)
                count=count+1
                
        print("More than 75 count " + str(count))
        
        count=0
        for c in all_accuracies:
            if(c<=50):
                count=count+1
                
        print("Less than 50 count " + str(count))
        
        count=0
        for c in all_accuracies:
            if(c>=60) and(c<70):
                count=count+1
                
        print(" 60 to 70 count " + str(count))
        
        count=0
        for c in all_accuracies:
            if(c>=70):
                count=count+1
                
        print("More than 70 count " + str(count))


def distribution_plot(experiment,num_feats_list,trans_names,selectors_list,results,label):
    all_accuracies=[]
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    for num_feats in num_feats_list:
            for tr_period in trans_names:
                for selector in selectors_list:
                    all_accuracies_rf=[]
                    all_accuracies_svm=[]
                    all_accuracies_xgb=[]
                    for result_type in results:
                        test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                        result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                        result_csv.drop("Unnamed: 0",axis=1,inplace=True)
            
                        all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                        all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                        all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())

                    all_accuracies.append(np.mean(all_accuracies_rf))
                    all_accuracies.append(np.mean(all_accuracies_xgb))
                    all_accuracies.append(np.mean(all_accuracies_svm))


    fig = make_subplots(rows=1, cols=2,horizontal_spacing = 0.05,
                    vertical_spacing= 0.02,shared_xaxes=True,subplot_titles=["Distribution of Model Accuracies", "Cumalative Distribution of Model Accuracies"])

    fig.update_layout(layout)
    fig.add_trace(go.Histogram(x=all_accuracies, nbinsx=10), row=1, col=1)
    fig.add_trace(go.Histogram(x=all_accuracies,nbinsx=10,histnorm='probability',cumulative_enabled=True), row=1, col=2)
    fig.update_layout(showlegend=False, xaxis_title="Model Accuracy", yaxis_title="Count")
    fig.update_layout(height=700, width=1200,
                   yaxis_zeroline=False, xaxis_zeroline=False, font=dict(color='white',
                     size=18,
                 ), title_text= "Adequate_testing results for models using " + label + " signal"
                   )
    fig.update_layout( legend=dict(font=dict(size=16)))
    fig.update_annotations(font=dict(size=18))
    fig.show()
    if not os.path.exists("Model_Trends"):
         os.mkdir("Model_Trends")
    fig.write_image("Model_Trends/anxiety_dist_"+experiment+".png")

    print(np.mean(all_accuracies))
    print(np.std(all_accuracies))
    
    count=0
    for c in all_accuracies:
        if(c>=75):
            print(c)
            count=count+1
            
    print("More than 75 count " + str(count))
    
    count=0
    for c in all_accuracies:
        if(c<=50):
            count=count+1
            
    print("Less than 50 count " + str(count))
    
    count=0
    for c in all_accuracies:
        if(c>=60) and(c<70):
            count=count+1
            
    print(" 60 to 70 count " + str(count))
    
    count=0
    for c in all_accuracies:
        if(c>=70):
            count=count+1
            
    print("More than 70 count " + str(count))

def distribution_plot_all(experiments,num_feats_list,trans_names,selectors_list,results):
    all_accuracies=[]
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    for experiment in experiments:
        for num_feats in num_feats_list:
                for tr_period in trans_names:
                    for selector in selectors_list:
                        all_accuracies_rf=[]
                        all_accuracies_svm=[]
                        all_accuracies_xgb=[]
                        for result_type in results:
                            test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                            result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                            result_csv.drop("Unnamed: 0",axis=1,inplace=True)
                
                            all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                            all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                            all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())
    
                        all_accuracies.append(np.mean(all_accuracies_rf))
                        all_accuracies.append(np.mean(all_accuracies_xgb))
                        all_accuracies.append(np.mean(all_accuracies_svm))


    fig = make_subplots(rows=1, cols=2,horizontal_spacing = 0.05,
                    vertical_spacing= 0.02,shared_xaxes=True,subplot_titles=["Distribution of Model Accuracies", "Cumalative Distribution of Model Accuracies"])

    
    fig.add_trace(go.Histogram(x=all_accuracies, nbinsx=10), row=1, col=1)
    fig.add_trace(go.Histogram(x=all_accuracies,nbinsx=10,histnorm='probability',cumulative_enabled=True), row=1, col=2)
    fig.update_layout(layout)
    fig.update_layout(showlegend=False, xaxis_title="Model Accuracy", yaxis_title="Count")
    fig.update_layout(height=700, width=1200,
                   yaxis_zeroline=False, xaxis_zeroline=False, font=dict(color='white',
                     size=16,
                 ), title_text= "Adequate_testing results for models with true labels" 
                   )
    fig.update_layout( legend=dict(font=dict(size=16)))
    fig.update_annotations(font=dict(size=18))
    fig.show()
    if not os.path.exists("Model_Trends"):
         os.mkdir("Model_Trends")
    fig.write_image("Model_Trends/anxiety_dist_all_experiment.png")

    print(np.mean(all_accuracies))
    print(np.std(all_accuracies))
    
    count=0
    for c in all_accuracies:
        if(c>=75):
            print(c)
            count=count+1
            
    print("More than 75 count " + str(count))

def best_accuracy_overall(experiment,num_feats_list,trans_names,selectors_list,results):
    
    name=""
    accuracy=0
    count=0
    
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    for num_feats in num_feats_list:
        for tr_period in trans_names:
            for selector in selectors_list:
                all_accuracies_rf=[]
                all_accuracies_svm=[]
                all_accuracies_xgb=[]
                for result_type in results:
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
                    all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                    all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                    all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())

                mean= np.mean(all_accuracies_rf)
                if(accuracy>=70):
                    count= count+1
                if(mean>accuracy):
                    accuracy=mean
                    name= "RF"+" "+selector +" "+tr_period +" "+str(num_feats)+" "
                mean= np.mean(all_accuracies_xgb)
                if(mean>accuracy):
                    accuracy=mean
                    name= "XGB"+" "+selector +" "+tr_period +" "+str(num_feats)+" "
                mean= np.mean(all_accuracies_svm)
                if(mean>accuracy):
                    accuracy=mean
                    name= "SVM"+" "+selector +" "+tr_period +" "+ str(num_feats)+" "

    print(experiment +" best accuracy overall \n")
    print(name)
    print(accuracy)
    print("Count over 70\n")
    print(count)


def best_trans_overall(experiment,num_feats_list,trans_names,selectors_list,results):
    
    name=""
    accuracy=0

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    for tr_period in trans_names:
        all_accuracies_rf=[]
        all_accuracies_svm=[]
        all_accuracies_xgb=[]
        for selector in selectors_list:
        
            for num_feats in num_feats_list:
                for result_type in results:
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
                    all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                    all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                    all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())
        all_accuracies= all_accuracies_rf+all_accuracies_xgb+all_accuracies_svm
        mean= np.mean(all_accuracies)
        if(mean>accuracy):
            accuracy=mean
            name= tr_period
           

    print(experiment +" best trans overall \n")
    print(name)
    print(accuracy)
    print("\n")
    
def best_selector_overall(experiment,num_feats_list,trans_names,selectors_list,results):
    
    name=""
    accuracy=0

    
    for selector in selectors_list:
        all_accuracies_rf=[]
        all_accuracies_svm=[]
        all_accuracies_xgb=[]
        for tr_period in trans_names:
        
        
            for num_feats in num_feats_list:
                for result_type in results:
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
                    all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                    all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                    all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())
        all_accuracies= all_accuracies_rf+all_accuracies_xgb+all_accuracies_svm
        mean= np.mean(all_accuracies)
        if(mean>accuracy):
            accuracy=mean
            name= selector 
           

    print(experiment +" best selector overall \n")
    print(name)
    print(accuracy)
    print("\n")
    

def best_selector_overall_withnumber(experiment,num_feats_list,trans_names,selectors_list,results):
    
    name=""
    accuracy=0

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    for selector in selectors_list:
        for num_feats in num_feats_list:
            all_accuracies_rf=[]
            all_accuracies_svm=[]
            all_accuracies_xgb=[]
            for tr_period in trans_names:
        
        
            
                for result_type in results:
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
                    all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                    all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                    all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())
            all_accuracies= all_accuracies_rf+all_accuracies_xgb+all_accuracies_svm
            mean= np.mean(all_accuracies)
            if(mean>accuracy):
                accuracy=mean
                name= selector + " " + str(num_feats)
           

    print(experiment +" best selector with number overall \n")
    print(name)
    print(accuracy)
    print("\n")

def plot_best(experiment,num_feats_list,trans_names,selectors_list,results,label):
    
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    selectors_list2=['Pearson Coefficient',
                    'Logistic Regression',
                    'Random Forest',
                    'Extra Tree',
                    'Chi-2',
                    'Sequential Forward',
                    'Sequential Backward',
                    'Mutual Info',
                    'Factor Analysis',
                    'PCA',
                    'Best All']
    name=["","","","",""]
    colors=["","","","",""]
    accuracy=[0,0,0,0,0,]
    std=[0,0,0,0,0,0]
    min_index= 0
    
    test_names=["","","","",""]
    
    for tr_period in trans_names:
    
        for num_feats in num_feats_list:
            
            
        
            for i,selector in enumerate(selectors_list):
                all_accuracies_rf=[]
                all_accuracies_svm=[]
                all_accuracies_xgb=[]
            
                for result_type in results:
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
                    all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                    all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                    all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())
            
            
                mean= np.mean(all_accuracies_rf)
                if(mean>accuracy[min_index]):
                    accuracy[min_index]=mean
                    name[min_index]= "</br>"+"RF"+"</br>"+selectors_list2[i] +"</br>"+tr_period +"</br>"+str(num_feats)+"features"
                    std[min_index]= np.std(all_accuracies_rf)
                    colors[min_index]=marker_colors[0]
                    test_names[min_index]=selector+'_'+tr_period+'_features'+str(num_feats)
                    
                    
                    min_value = min(accuracy)
                    min_index=accuracy.index(min_value)
                    
                
                
                mean= np.mean(all_accuracies_xgb)
                if(mean>accuracy[min_index]):
                    accuracy[min_index]=mean
                    name[min_index]= "</br>"+"XGB"+"</br>"+selectors_list2[i]  +"</br>"+tr_period +"</br>"+str(num_feats)+"features"
                    std[min_index]= np.std(all_accuracies_xgb)
                    colors[min_index]=marker_colors[1]
                    test_names[min_index]=selector+'_'+tr_period+'_features'+str(num_feats)
                    
                    min_value = min(accuracy)
                    min_index=accuracy.index(min_value)
                    
                
                
                mean= np.mean(all_accuracies_svm)
                
                if(mean>accuracy[min_index]):
                    accuracy[min_index]=mean
                    name[min_index]= "</br>"+"SVM"+"</br>"+selectors_list2[i]  +"</br>"+tr_period +"</br>"+ str(num_feats)+"features"
                    std[min_index]= np.std(all_accuracies_svm)
                    colors[min_index]=marker_colors[2]
                    test_names[min_index]=selector+'_'+tr_period+'_features'+str(num_feats)
                    
                    min_value = min(accuracy)
                    min_index=accuracy.index(min_value)
                    
                
                
           
                
                
                
    fig = go.Figure([go.Bar(x=name, y=accuracy,
    error_y=dict(type='data', array=std),
    # marker_color=marker_colors[0],
    text=round_up_list(accuracy),
    textposition='inside',
    insidetextanchor='middle',
    marker_color=colors
    )])
    
     
    fig.update_layout(layout)     
    fig.update_traces( marker_line_width=2)

    fig.update_layout(height=600, width=1100, xaxis_title= "Models",
    yaxis_title="Accuracy",font=dict(color='white',
              size=16,  # Set the font size here
              
        ), title_text= "Top 5 "+ label + " models based on Adequate_testing")
    fig.update_annotations(font=dict(size=20))
    fig.show()
    fig.update_layout(legend=dict(
        font=dict(size=20),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        ))
        
    if not os.path.exists("Model_Trends"):
        os.mkdir("Model_Trends")
    fig.write_image("Model_Trends/"+experiment+".png")
    
    
    features_used2(experiment, test_names,label)
        


def plot_best_data_split(experiment,num_feats_list,trans_names,selectors_list,results,label):
    
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    selectors_list2=['Pearson Coefficient',
                    'Logistic Regression',
                    'Random Forest',
                    'Extra Tree',
                    'Chi-2',
                    'Sequential Forward',
                    'Sequential Backward',
                    'Mutual Info',
                    'Factor Analysis',
                    'PCA',
                    'Best All']
    
    
    for result_type in results:
        name=["","","","",""]
        colors=["","","","",""]
        accuracy=[0,0,0,0,0,]
        std=[0,0,0,0,0,0]
        min_index= 0
        
        test_names=["","","","",""]
        for tr_period in trans_names:
        
            for num_feats in num_feats_list:
            
            
        
                for i,selector in enumerate(selectors_list):
                    all_accuracies_rf=[]
                    all_accuracies_svm=[]
                    all_accuracies_xgb=[]
            
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
                    all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                    all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                    all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())
            
            
                    mean= np.mean(all_accuracies_rf)
                    if(mean>accuracy[min_index]):
                        accuracy[min_index]=mean
                        name[min_index]= "</br>"+"RF"+"</br>"+selectors_list2[i] +"</br>"+tr_period +"</br>"+str(num_feats)+"features"
                        std[min_index]= np.std(all_accuracies_rf)
                        colors[min_index]=marker_colors[0]
                        test_names[min_index]=selector+'_'+tr_period+'_features'+str(num_feats)
                        
                        
                        min_value = min(accuracy)
                        min_index=accuracy.index(min_value)
                        
                    
                    
                    mean= np.mean(all_accuracies_xgb)
                    if(mean>accuracy[min_index]):
                        accuracy[min_index]=mean
                        name[min_index]= "</br>"+"XGB"+"</br>"+selectors_list2[i]  +"</br>"+tr_period +"</br>"+str(num_feats)+"features"
                        std[min_index]= np.std(all_accuracies_xgb)
                        colors[min_index]=marker_colors[1]
                        test_names[min_index]=selector+'_'+tr_period+'_features'+str(num_feats)
                        
                        min_value = min(accuracy)
                        min_index=accuracy.index(min_value)
                        
                    
                    
                    mean= np.mean(all_accuracies_svm)
                    
                    if(mean>accuracy[min_index]):
                        accuracy[min_index]=mean
                        name[min_index]= "</br>"+"SVM"+"</br>"+selectors_list2[i]  +"</br>"+tr_period +"</br>"+ str(num_feats)+"features"
                        std[min_index]= np.std(all_accuracies_svm)
                        colors[min_index]=marker_colors[2]
                        test_names[min_index]=selector+'_'+tr_period+'_features'+str(num_feats)
                        
                        min_value = min(accuracy)
                        min_index=accuracy.index(min_value)
                    
                
                
           
                
                
                
        fig = go.Figure([go.Bar(x=name, y=accuracy,
        error_y=dict(type='data', array=std),
        # marker_color=marker_colors[0],
        text=round_up_list(accuracy),
        textposition='inside',
        insidetextanchor='middle',
        marker_color=colors
        )])
        
        fig.update_layout(layout)     
        fig.update_traces( marker_line_width=2)
        
        if(result_type=="random"):
            result_type="Random"
    
        fig.update_layout(height=600, width=1100, xaxis_title= "Models",
        yaxis_title="Accuracy",title_text= "Best 5 " +label+ " models tested using " +result_type , font=dict(
                  color='white',size=18,  # Set the font size here
    
            ))
        fig.update_annotations(font=dict(size=18))
        fig.show()
        fig.update_layout(legend=dict(
            font=dict(size=20),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            ))
            
        if not os.path.exists("Model_Trends4"):
            os.mkdir("Model_Trends4")
        fig.write_image("Model_Trends4/"+experiment+"_"+result_type+".png")
        
        



def features_used(experiment,num_feats_list,trans_names,selectors_list,results,label):
    feature_list=[]
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    for tr_period in trans_names:
    
        for num_feats in num_feats_list:
            for i,selector in enumerate(selectors_list):
                test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                filename= experiment+"/"+test_name+"/" +"features_used_with_"+test_name+".txt"
                with open(filename) as f:
                    for i,line in enumerate(f):
                        if(i>1):
                            
                            line_split=line.split()
                            feature_list.append(line_split[1])
                    
    

    unique, counts = np.unique(feature_list, return_counts=True)
    
    df = pd.DataFrame(unique, columns=['Feature Name'])
    df['Count']=counts
    df.sort_values(by=['Count'], inplace=True, ascending= False)
    df_short= df.head(10)
    fig = px.bar(df_short, x='Feature Name', y='Count')
    fig.update_layout(layout)
    fig.update_layout(height=600, width=1100, title= "Best 10 Features selected across all "+label +" models",font=dict(
              color='white',size=18,  # Set the font size here

        ))
    fig.show()
    if not os.path.exists("Model_Trends"):
        os.mkdir("Model_Trends")
    fig.write_image("Model_Trends/"+experiment+"_features.png")
    

def features_used2(experiment,test_names,label):
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    feature_list=[]
    for test_name in test_names:
        filename= experiment+"/"+test_name+"/" +"features_used_with_"+test_name+".txt"
        with open(filename) as f:
                for i,line in enumerate(f):
                    if(i>1):
                        
                        line_split=line.split()
                        feature_list.append(line_split[1])
                


    unique, counts = np.unique(feature_list, return_counts=True)
    
    df = pd.DataFrame(unique, columns=['Feature Name'])
    df['Count']=counts
    df.sort_values(by=['Count'], inplace=True, ascending= False)
    df_short= df.head(5)
    fig = px.bar(df_short, x='Feature Name', y='Count')
    fig.update_layout(layout)
    fig.update_layout(height=600, width=1100, title= "Best 5 Features selected across adequately tested top 5 "+ label+" models",font=dict(
              color='white',size=18,  # Set the font size here
    
        ))
    fig.show()
    if not os.path.exists("Model_Trends"):
        os.mkdir("Model_Trends")
    fig.write_image("Model_Trends/"+experiment+"_features_best_5models.png")
        


def plot_std_bar_graph_subplots(experiment,num_feats_list,trans_names,selectors_list,results):
      
    
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    
    
    for num_feats in num_feats_list:
        for tr_period in trans_names:
            
            rf_mean_list=[]
            rf_std_list=[]
            xgb_mean_list=[]
            xgb__std_list=[]
            svm_mean_list=[]
            svm_std_list=[]
            fig = go.Figure()
            for selector in selectors_list:
                all_accuracies_rf=[]
                all_accuracies_svm=[]
                all_accuracies_xgb=[]
                for result_type in results:
                    test_name= selector+'_'+tr_period+'_features'+str(num_feats)
                    result_csv= pd.read_csv(experiment+"/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
                    result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
                    all_accuracies_rf= all_accuracies_rf+ percentage_list(result_csv["rf_accuracy"].tolist())
                    all_accuracies_xgb= all_accuracies_xgb+ percentage_list(result_csv["xgb_accuracy"].tolist())
                    all_accuracies_svm= all_accuracies_svm+ percentage_list(result_csv["svm_accuracy"].tolist())
            
            
                
                rf_mean_list.append(np.mean(all_accuracies_rf))
                xgb_mean_list.append(np.mean(all_accuracies_xgb))
                svm_mean_list.append(np.mean(all_accuracies_svm))

                
                rf_std_list.append(np.std(all_accuracies_rf))
                xgb__std_list.append(np.std(all_accuracies_xgb))
                svm_std_list.append(np.std(all_accuracies_svm))
            

            fig.add_trace(go.Bar(
                    name="Random Forest",
                    x=selectors_list, y=rf_mean_list,
                    error_y=dict(type='data', array=rf_std_list),
                    marker_color=marker_colors[0],
                    text=round_up_list(rf_mean_list),
                    textposition='inside',
                    insidetextanchor='middle'
                    ))
            
            fig.add_trace(go.Bar(
                    name="XGB ",
                    x=selectors_list, y=xgb_mean_list,
                    error_y=dict(type='data', array=xgb__std_list),
                    marker_color=marker_colors[1],
                    text=round_up_list(xgb_mean_list),
                    textposition='inside',
                    insidetextanchor='middle'
                    
                    ))
            
            fig.add_trace(go.Bar(
                    name="SVM",
                    x=selectors_list, y=svm_mean_list,
                    error_y=dict(type='data', array=svm_std_list),
                    
                    marker_color=marker_colors[2],
                    text=round_up_list(svm_mean_list),
                    textposition='inside',
                    insidetextanchor='middle'
                    ))
            fig.update_traces( marker_line_width=2)
            fig.update_layout(layout)
            fig.update_layout(height=1700, width=2000,
                      yaxis_zeroline=False, xaxis_zeroline=False, 
                      title_text="</b>"+"Transitional Period "+ tr_period.replace("trans", "")+ " seconds plots for " +str(num_feats)+" features"+"</b>"
                      )
        
            fig.update_layout(
                      yaxis_zeroline=False, xaxis_zeroline=False, 
                      font=dict(
                         color='white', size=18,  # Set the font size here

                    ))
            fig.update_annotations(font=dict(size=25))
            fig.show()
            fig.update_layout(legend=dict(
                font=dict(size=30),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
                ))
        
            if not os.path.exists("Model_Trends2"):
                os.mkdir("Model_Trends2")
            if not os.path.exists("Model_Trends2/"+experiment):
                os.mkdir("Model_Trends2/"+experiment)
            fig.write_image("Model_Trends2/"+experiment+"/"+ str(num_feats)+"features_"+tr_period+".png")



