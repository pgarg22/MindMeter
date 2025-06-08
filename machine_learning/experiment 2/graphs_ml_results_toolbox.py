#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 23:08:46 2023

@author: octopusphoenix
"""



from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd




def push_heatmap(z,modname,directory):
    
    x = ['No Anixety', 'Anxiety']
    y =  ['No Anixety', 'Anxiety']
    
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]
    
    
    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='GnBu')

    # add title
    fig.update_layout(title_text='<i><b>'+modname+' matrix</b></i>',
                      #xaxis = dict(title='x'),
                      #yaxis = dict(title='x')
                     )
    
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    
    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))
    
    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()
    
    if not os.path.exists(directory+"Confusion_matrices"):
        os.mkdir(directory+"Confusion_matrices")
    fig.write_image(directory+"Confusion_matrices/"+modname+".png")

    





def push_viz_scatter_subplots2(row,col, df,title,subtitles,ytitle,mode,symbol,colors,directory):
    
    
    fig = make_subplots(rows=row, cols=col,horizontal_spacing = 0,
                    vertical_spacing= 0.20,
                    subplot_titles=subtitles, y_title=ytitle,shared_yaxes=True)
    c=1
    for i in range(1,(row)+1):
        for j in range(1,(col)+1):
            fig.append_trace(go.Scatter(
            x=df.iloc[:, 0],
            y=df.iloc[:, c],
            name=df.columns[c],
            mode=mode,
            marker=dict(
                symbol=symbol[j-1], line=dict(width=2, color="DarkSlateGrey"),
               color=colors[j-1], #set color equal to a variable
          )
              
            
            ), row=i, col=j)
            c=c+1
            if(c==df.shape[1]):
                break;
        if(c==df.shape[1]):
            break;
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                
                      yaxis_zeroline=False, xaxis_zeroline=False, 
                      )
   
    fig.show()

    if not os.path.exists(directory+"Participant_Video_Trends"):
        os.mkdir(directory+"Participant_Video_Trends")
    fig.write_image(directory+"Participant_Video_Trends/"+title+".png")
    return(fig)
    
def push_ml_results_trans(directory,test_name):
        
    """
    ============================================================================================================
    Loading data
    ============================================================================================================
    """
    
    df_metadata= pd.read_csv("processed_files/Metadata.csv")
    cols=["Participant ID","Beck Anxiety","Hamilton Anxiety"]
    df_metadata=df_metadata[cols]
    
    
    colors= ["green", "orange", "red","blue"]
    ytitle="Values"
    mode= 'markers+lines'
    symbols= ["circle","diamond","cross","arrow"]
    
    """
    ============================================================================================================
    Graph 1
    ============================================================================================================
    """
    
    df_participant_results_trans= pd.read_csv(directory+test_name+"_Participant_split_results.csv")
    df_participant_results_trans.rename(columns = {'Unnamed: 0':'Participant ID'}, inplace = True)
    df_participant_results_trans['Participant ID']=df_participant_results_trans['Participant ID'] +1
    
    new_df_trans = pd.merge(df_participant_results_trans, df_metadata,  how='left', left_on=['Participant ID'], right_on = ['Participant ID'])   
    
    graph_title= test_name+ "_Participant_split_results"
    
    subtitles= ['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score', 
                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score', 
                'svm_accuracy','svm_precision', 'svm_recall', 'svm_f1score', 
                'Beck Anxiety','Hamilton Anxiety']
    
    cols=["Participant ID","rf_accuracy","rf_precision", "rf_recall", "rf_f1score", 
                "xgb_accuracy", "xgb_precision", "xgb_recall", "xgb_f1score", 
                "svm_accuracy","svm_precision", "svm_recall", "svm_f1score", 
                "Beck Anxiety","Hamilton Anxiety"]
    
    new_df_trans=new_df_trans[cols] 
    push_viz_scatter_subplots2(4,4,new_df_trans,graph_title,subtitles ,ytitle,mode ,symbols,colors,directory)
    
    """
    ============================================================================================================
    Graph 2
    ============================================================================================================
    """
    
    graph_title= test_name+ "_Video_split_results"
    subtitles= ['rf_accuracy','xgb_accuracy', 'svm_accuracy']
    
    df_video_results_trans= pd.read_csv(directory+test_name+"_Video_split_results.csv")
    
    df_video_results_trans.rename(columns = {'Unnamed: 0':'Video'}, inplace = True)
    df_video_results_trans['Video']=df_video_results_trans['Video'] +1
    cols=["Video","rf_accuracy","xgb_accuracy","svm_accuracy"]
    
    
    df_video_results_trans=df_video_results_trans[cols]    
    push_viz_scatter_subplots2(1,3,df_video_results_trans,graph_title,subtitles ,ytitle,mode ,symbols,colors,directory)
    
    
    """
    ============================================================================================================
    Graph 3
    ============================================================================================================
    """
    
    graph_title= test_name+ "_random_split_results"
    subtitles= ['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score', 
                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score', 
                'svm_accuracy','svm_precision', 'svm_recall', 'svm_f1score']
    
    df_random_results_trans= pd.read_csv(directory+test_name+"_random_split_results.csv")
    
    df_random_results_trans.rename(columns = {'Unnamed: 0':'Index'}, inplace = True)
    df_random_results_trans['Index']=df_random_results_trans['Index'] +1
    cols=["Index","rf_accuracy","rf_precision", "rf_recall", "rf_f1score", 
                "xgb_accuracy", "xgb_precision", "xgb_recall", "xgb_f1score", 
                "svm_accuracy","svm_precision", "svm_recall", "svm_f1score"]
    df_random_results_trans=df_random_results_trans[cols] 
    
    df_random_results_trans=df_random_results_trans[cols]    
    push_viz_scatter_subplots2(4,3,df_random_results_trans,graph_title,subtitles ,ytitle,mode ,symbols,colors,directory)
    
    
    

    
    