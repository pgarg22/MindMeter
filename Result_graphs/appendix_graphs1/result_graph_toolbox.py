#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:14:42 2023

@author: octopusphoenix
"""
import pandas as pd
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots


marker_colors=["#bdcf32","#edbf33","#b33dc6"]

def round_up_list(my_list):
    rounded_list= [ '%.1f' % elem for elem in my_list ]
    return(rounded_list)

def percentage_list(my_list):
    percent_list= [ 100*elem for elem in my_list]
    return(percent_list)

def plot_std_bar_graph(selectors_list,tr_period,num_feats,result_type,metric):
    
    
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        )
    rf_mean_list=[]
    rf_std_list=[]
    xgb_mean_list=[]
    xgb__std_list=[]
    svm_mean_list=[]
    svm_std_list=[]
    for selector in selectors_list:
        test_name= selector+'_'+tr_period+'_features'+str(num_feats)
        result_csv= pd.read_csv("test_results/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
        result_csv.drop("Unnamed: 0",axis=1,inplace=True)
        
        result_csv_mean=result_csv.mean()
        result_csv_std=result_csv.std()
        
        rf_mean_list.append(result_csv_mean['rf_'+metric])
        xgb_mean_list.append(result_csv_mean['xgb_'+metric])
        svm_mean_list.append(result_csv_mean['svm_'+metric])
        
        rf_std_list.append(result_csv_std['rf_'+metric])
        xgb__std_list.append(result_csv_std['xgb_'+metric])
        svm_std_list.append(result_csv_std['svm_'+metric])
    
    rf_mean_list= percentage_list(rf_mean_list)
    rf_std_list= percentage_list(rf_std_list)
    xgb_mean_list= percentage_list(xgb_mean_list)
    xgb__std_list= percentage_list(xgb__std_list)
    svm_mean_list= percentage_list(svm_mean_list)
    svm_std_list= percentage_list(svm_std_list)
    
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Bar(
            name="Random Forest "+ metric,
            x=selectors_list, y=rf_mean_list,
            error_y=dict(type='data', array=rf_std_list),
            marker_color=marker_colors[0],
            text=round_up_list(rf_mean_list),
            textposition='inside',
            insidetextanchor='middle'
            ))
    
    fig.add_trace(go.Bar(
            name="XGB  "+ metric,
            x=selectors_list, y=xgb_mean_list,
            error_y=dict(type='data', array=xgb__std_list),
            marker_color=marker_colors[1],
            text=round_up_list(xgb_mean_list),
            textposition='inside',
            insidetextanchor='middle'
            ))
    
    fig.add_trace(go.Bar(
            name="SVM  "+ metric,
            x=selectors_list, y=svm_mean_list,
            error_y=dict(type='data', array=svm_std_list),
            marker_color=marker_colors[2],
            text=round_up_list(svm_mean_list),
            textposition='inside',
            insidetextanchor='middle'
            ))
    fig.update_layout(barmode='group')
    fig.update_layout(height=1200, width=2000,
                  yaxis_zeroline=False, xaxis_zeroline=False, 
                  title_text="<b>""ECG Signal: " +metric.capitalize()+ " plot for " + result_type +" split using "+str(num_feats)+" features and transitional period "+ tr_period +" seconds</b>"
                  )
    fig.show()

    if not os.path.exists("Model_Trends"):
        os.mkdir("Model_Trends")
    if not os.path.exists("Model_Trends/"+metric):
        os.mkdir("Model_Trends/"+metric)
    fig.write_image("Model_Trends/"+metric+"/"+result_type+str(num_feats)+"features_"+tr_period+"_"+metric+".png")



def plot_std_bar_graph_subplots(selectors_list,tr_period,num_feats,metric_list, result_type):
      
    
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        )
    
    count=1
    fig = make_subplots(rows=4, cols=1,horizontal_spacing = 0,
                    vertical_spacing= 0.02,shared_xaxes=True,subplot_titles=metric_list)
    for metric in metric_list:
        rf_mean_list=[]
        rf_std_list=[]
        xgb_mean_list=[]
        xgb__std_list=[]
        svm_mean_list=[]
        svm_std_list=[]
        for selector in selectors_list:
            test_name= selector+'_'+tr_period+'_features'+str(num_feats)
            result_csv= pd.read_csv("test_results/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
            result_csv.drop("Unnamed: 0",axis=1,inplace=True)
            
            result_csv_mean=result_csv.mean()
            result_csv_std=result_csv.std()
            
            rf_mean_list.append(result_csv_mean['rf_'+metric])
            xgb_mean_list.append(result_csv_mean['xgb_'+metric])
            svm_mean_list.append(result_csv_mean['svm_'+metric])
            
            rf_std_list.append(result_csv_std['rf_'+metric])
            xgb__std_list.append(result_csv_std['xgb_'+metric])
            svm_std_list.append(result_csv_std['svm_'+metric])
        
        rf_mean_list= percentage_list(rf_mean_list)
        rf_std_list= percentage_list(rf_std_list)
        xgb_mean_list= percentage_list(xgb_mean_list)
        xgb__std_list= percentage_list(xgb__std_list)
        svm_mean_list= percentage_list(svm_mean_list)
        svm_std_list= percentage_list(svm_std_list)
        fig.add_trace(go.Bar(
                name="Random Forest "+ metric,
                x=selectors_list, y=rf_mean_list,
                error_y=dict(type='data', array=rf_std_list),
                marker_color=marker_colors[0],
                text=round_up_list(rf_mean_list),
                textposition='inside',
                insidetextanchor='middle'
                ), row=count, col=1)
        
        fig.add_trace(go.Bar(
                name="XGB  "+ metric,
                x=selectors_list, y=xgb_mean_list,
                error_y=dict(type='data', array=xgb__std_list),
                marker_color=marker_colors[1],
                text=round_up_list(xgb_mean_list),
                textposition='inside',
                insidetextanchor='middle'
                
                ), row=count, col=1)
        
        fig.add_trace(go.Bar(
                name="SVM  "+ metric,
                x=selectors_list, y=svm_mean_list,
                error_y=dict(type='data', array=svm_std_list),
                
                marker_color=marker_colors[2],
                text=round_up_list(svm_mean_list),
                textposition='inside',
                insidetextanchor='middle'
                ), row=count, col=1)
        count=count+1
    fig.update_traces( marker_line_width=2)
    
    if(result_type=="random"):
        result_type="Random"
    fig.update_layout(height=1200, width=2000,
                  yaxis_zeroline=False, xaxis_zeroline=False, 
                  title_text="</b>"+"Transitional Period "+ tr_period.replace("trans", "")+ " seconds plots for " + result_type +" split using "+str(num_feats)+" features for transitional period in both beginning and end of video"+"</b>"
                  )
    
    fig.update_layout(height=1200, width=2000,
                  yaxis_zeroline=False, xaxis_zeroline=False, 
                  )
    fig.show()
    
    if not os.path.exists("Model_Trends2"):
        os.mkdir("Model_Trends2")
    fig.write_image("Model_Trends2/"+result_type+str(num_feats)+"features_"+tr_period+".png")



def plot_std_bar_graph_subplots_trans_period(selectors_list,tr_names,num_feats,result_type,metric):
      
    
    layout = go.Layout(
        autosize=False,
        width=2000,
        height=1000,
        )

    col_number=1
    row_number=1
    fig = make_subplots(rows=4, cols=1,horizontal_spacing = 0.05,
                    vertical_spacing= 0.05,shared_xaxes=True,subplot_titles=tr_names)
    
    
    
    legend=True
    
    for i,tr_period in enumerate(tr_names):
        rf_mean_list=[]
        rf_std_list=[]
        xgb_mean_list=[]
        xgb__std_list=[]
        svm_mean_list=[]
        svm_std_list=[]
        for selector in selectors_list:
            test_name= selector+'_'+tr_period+'_features'+str(num_feats)
            result_csv= pd.read_csv("test_results/"+test_name+"/" +test_name+'_'+result_type+'_split_results.csv')
            result_csv.drop("Unnamed: 0",axis=1,inplace=True)
            
            result_csv_mean=result_csv.mean()
            result_csv_std=result_csv.std()
            
            rf_mean_list.append(result_csv_mean['rf_'+metric])
            xgb_mean_list.append(result_csv_mean['xgb_'+metric])
            svm_mean_list.append(result_csv_mean['svm_'+metric])
            
            rf_std_list.append(result_csv_std['rf_'+metric])
            xgb__std_list.append(result_csv_std['xgb_'+metric])
            svm_std_list.append(result_csv_std['svm_'+metric])
        
        rf_mean_list= percentage_list(rf_mean_list)
        rf_std_list= percentage_list(rf_std_list)
        xgb_mean_list= percentage_list(xgb_mean_list)
        xgb__std_list= percentage_list(xgb__std_list)
        svm_mean_list= percentage_list(svm_mean_list)
        svm_std_list= percentage_list(svm_std_list)
        
        fig.add_trace(go.Bar(
                name="Random Forest "+ metric,
                x=selectors_list, y=rf_mean_list,
                error_y=dict(type='data', array=rf_std_list),
                marker_color=marker_colors[0],
                text=round_up_list(rf_mean_list),
                textposition='inside',
                insidetextanchor='middle',
                showlegend=legend
                ), row=row_number, col=col_number)
        
        fig.add_trace(go.Bar(
                name="XGB  "+ metric,
                x=selectors_list, y=xgb_mean_list,
                error_y=dict(type='data', array=xgb__std_list),
                marker_color=marker_colors[1],
                text=round_up_list(xgb_mean_list),
                textposition='inside',
                insidetextanchor='middle',
                showlegend=legend
                
                ), row=row_number, col=col_number)
        
        fig.add_trace(go.Bar(
                name="SVM  "+ metric,
                x=selectors_list, y=svm_mean_list,
                error_y=dict(type='data', array=svm_std_list),
                
                marker_color=marker_colors[2],
                text=round_up_list(svm_mean_list),
                textposition='inside',
                insidetextanchor='middle',
                showlegend=legend
                ), row=row_number, col=col_number)
        if(col_number==1):
            row_number=row_number+1
            col_number=1
        else:
            col_number=col_number+1
        
        legend=False
    
    fig.update_traces( marker_line_width=2)
    
    fig.update_layout(height=1200, width=2000,
                  yaxis_zeroline=False, xaxis_zeroline=False, 
                  title_text="<b>"+metric.capitalize()+ " plots for " + result_type +" split using "+str(num_feats)+" features for transitional period in both beginning and end of video"+ "</b>"
                  )
    fig.show()
    
    if not os.path.exists("Model_Trends3"):
        os.mkdir("Model_Trends3")
    if not os.path.exists("Model_Trends3/"+metric):
        os.mkdir("Model_Trends3/"+metric)
    fig.write_image("Model_Trends3/"+metric+"/"+result_type+str(num_feats)+"features_"+metric+".png")

    
    
   