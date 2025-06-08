#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:27:52 2023

@author: octopusphoenix
"""



from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import numpy as np
import plotly.express as px



colors_scale_options= [[(0.00, "#e6f69d"),   (0.11, "#e6f69d"),
                         (0.11, "#aadea7"), (0.24, "#aadea7"),
                         (0.24, "#64c2a6"),  (0.4, "#64c2a6"),
                         (0.4, "#2d87bb"),  (1.00, "#2d87bb")],
                        [(0.00, "#fef001"),   (0.125, "#fef001"),
                         (0.125, "#fd9a01"), (0.25, "#fd9a01"),
                         (0.25, "#fd6104"), (0.41, "#fd6104"),
                         (0.41, "#F00505"),  (1.00, "#ff2c05")], ] 



"""
============================================================================================================
Function to calculate BMI
============================================================================================================
"""
def get_bmi(height,weight):
    return np.round(((weight*4535.9237)/(height*height))) #height in cm and weight in lbs

"""
============================================================================================================
"""


"""
============================================================================================================
Function to plot a scatter plot in plotly
============================================================================================================
"""

def push_viz_scatter(Number, df,title,ytitle,mode,symbol,range_max,tick_text):
    
    
    fig = go.Figure()
    cs_place=1.0
    barlength=378
    for i in range(1,Number+1):
        
        
    
        fig.add_trace(go.Scatter(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        mode=mode,
        marker=dict(
            symbol=symbol[i-1], line=dict(width=2, color="DarkSlateGrey"),
           size=df.iloc[:, i]+15,
           color=df.iloc[:, i], #set color equal to a variable
           colorscale=colors_scale_options[i-1],
           colorbar=dict( x= cs_place, title=df.columns[i],
           tickvals=[4,12,20,40],
           ticktext=tick_text[i-1],
           lenmode="pixels", len=barlength, ),
            # one of plotly colorscales
           showscale=True,
           cmax=range_max[i-1],
           cmin=0)))
        cs_place=cs_place+0.08
        if(barlength==378):
            barlength=336
        else:
            barlength=378
        
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                      yaxis_title=ytitle,
                      legend_title="Legend",
                      yaxis_zeroline=False, xaxis_zeroline=False,
                      font=dict(
                        size=30,
                    ))
    fig.show()
    
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")


"""
============================================================================================================
"""


"""
============================================================================================================
Function to plot a scatter plot  as subplots in plotly
============================================================================================================
"""

def push_viz_scatter_subplots(Number, df,title, subtitles,ytitle,mode,symbol,range_max,tick_text):
    
    
    fig = make_subplots(rows=1, cols=Number,horizontal_spacing = 0,
                    vertical_spacing= 0.20,
                    subplot_titles=subtitles,shared_yaxes=True)
    
    cs_place=1.0
    barlength= 472.5
    for i in range(1,Number+1):
        fig.append_trace(go.Scatter(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        mode=mode,
        marker=dict(
            symbol=symbol[i-1], line=dict(width=2, color="DarkSlateGrey"),
           size=df.iloc[:, i]+15,
           color=df.iloc[:, i], #set color equal to a variable
           colorscale=colors_scale_options[i-1],
           colorbar=dict( x= cs_place, title=df.columns[i],
           tickvals=[4,12,20,40],
           ticktext=tick_text[i-1],
           lenmode="pixels", len=barlength, ),
            # one of plotly colorscales
           showscale=True,
           cmax=range_max[i-1],
           cmin=0)
        
        ), row=1, col=i)
        cs_place=cs_place+0.12
        if(barlength==472.5):
            barlength=420
        else:
            barlength=472.5
        
    
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                      yaxis_title=ytitle,
                      yaxis_zeroline=False, xaxis_zeroline=False, font=dict(
                          size=20,  # Set the font size here

                    ))
    fig.update_layout( legend=dict(font=dict(size=16)))
    fig.update_annotations(font=dict(size=20))
                      
    fig.show()

    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")
    
"""
============================================================================================================
"""







"""
============================================================================================================
Function to plot a bar plot in plotly
============================================================================================================
"""
def push_viz_bar(Number, df,title, subtitles,ytitle,range_max,tick_text):
    
    
    fig = go.Figure()
    cs_place=1.0
    barlength=378
    for i in range(1,Number+1):
        
        df= df.sort_values(by=df.columns[i])
    
        fig.add_trace(go.Bar(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        marker=dict( 
           color=df.iloc[:, i], #set color equal to a variable
           colorscale=colors_scale_options[i-1],
           colorbar=dict( x= cs_place, title=df.columns[i],
           tickvals=[4,12,20,40],
           ticktext=tick_text[i-1],
           lenmode="pixels", len=barlength, ),
            # one of plotly colorscales
           showscale=True,
           cmax=range_max[i-1],
           cmin=0)))
        cs_place=cs_place+0.08
        if(barlength==378):
            barlength=336
        else:
            barlength=378

        
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                      yaxis_title=ytitle,
                      legend_title="Legend",
                  yaxis_zeroline=False, xaxis_zeroline=False,
                  xaxis={ 'categoryorder':'array', 'categoryarray':['<10', '10-20', '20-30','30-40', '40-50','50-60','60+']}
                  , font=dict(
                      size=16,  # Set the font size here

                ))
    fig.update_annotations(font=dict(size=18))
    fig.show()
    
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")
    

"""
============================================================================================================
"""


"""
============================================================================================================
Function to plot a bar plots as subplots in plotly
============================================================================================================
"""
    
def push_viz_bar_subplots(Number, df,title, subtitles,ytitle,range_max,tick_text):
    
    cs_place=1.0
    barlength=378
    fig = make_subplots(rows=1, cols=Number,horizontal_spacing = 0.15,
                    vertical_spacing= 0.20,
                    subplot_titles=subtitles)
    for i in range(1,Number+1):
        df= df.sort_values(by=df.columns[i])
        fig.append_trace(go.Bar(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        marker=dict(
           
           color=df.iloc[:, i], #set color equal to a variable
           colorscale=colors_scale_options[i-1],
           colorbar=dict( x= cs_place, title=df.columns[i],
           tickvals=[4,12,20,40],
           ticktext=tick_text[i-1],
           lenmode="pixels", len=barlength, ),
            # one of plotly colorscales
           showscale=True,
           cmax=range_max[i-1],
           cmin=0)), row=1, col=i)

        cs_place=cs_place+0.08
        if(barlength==378):
            barlength=336
        else:
            barlength=378
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                  xaxis_title= df.columns[0],
                  xaxis={'categoryorder':'array', 'categoryarray':['<10', '10-20', '20-30','30-40', '40-50','50-60','60+']},
                  yaxis_zeroline=False, xaxis_zeroline=False, 
                  )
    fig.show()
    
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")

"""
============================================================================================================
"""




    
    
    