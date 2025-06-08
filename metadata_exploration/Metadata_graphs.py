#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 23:10:06 2022

@author: octopusphoenix
"""

from metadata_toolbox import push_viz_scatter , push_viz_bar, push_viz_scatter_subplots,push_viz_bar_subplots
import pandas as pd


##############################################################################################

demographic_df= pd.read_csv('Metadata.csv')
# Graph in order the data collected from participants(Participant ID)
push_viz_scatter_subplots(2, 
                 demographic_df.loc[:, ["Participant ID","Beck Anxiety", "Hamilton Anxiety"]],
                 "Anxiety Distributions",
                 ["Beck Anxiety", "Hamilton Anxiety"],
                 "Anxiety Level",
                 "markers+lines",
                 ["diamond","cross"],
                 [63,56],
                 [["Minimum</br></br>(0-7)","Mild</br></br>(8-15)","Moderate</br></br>(16-25)","Severe</br></br>(26-63)"],
                  ["Minimum</br></br>(0-7)","Mild</br></br>(8-14)","Moderate</br></br>(15-23)","Severe</br></br>(24-56)"]])


##############################################################################################



# Graph in order of Age Scatter'
push_viz_scatter_subplots(2, 
                 demographic_df.loc[:, ["Age","Beck Anxiety", "Hamilton Anxiety"]],
                 "Age",
                 ["Beck Anxiety", 
                  "Hamilton Anxiety"],
                 "Anxiety Level",
                 "markers",
                 ["diamond","cross"],
                 [63,56],
                 [["Minimum</br></br>(0-7)","Mild</br></br>(8-15)","Moderate</br></br>(16-25)","Severe</br></br>(26-63)"],
                  ["Minimum</br></br>(0-7)","Mild</br></br>(8-14)","Moderate</br></br>(15-23)","Severe</br></br>(24-56)"]])


##############################################################################################


# Graph in order of Age Bar '
push_viz_bar(2,
             demographic_df.loc[:, ["AgeRange","Beck Anxiety", "Hamilton Anxiety"]],
             "Age_bar",
             ["Beck Anxiety", "Hamilton Anxiety"],
             "Anxiety Level",
             [63,56],
                 [["Minimum</br></br>(0-7)","Mild</br></br>(8-15)","Moderate</br></br>(16-25)","Severe</br></br>(26-63)"],
                  ["Minimum</br></br>(0-7)","Mild</br></br>(8-14)","Moderate</br></br>(15-23)","Severe</br></br>(24-56)"]])

##############################################################################################



# Graph in order of Gender '
push_viz_scatter_subplots(2, 
                 demographic_df.loc[:, ["Gender","Beck Anxiety", "Hamilton Anxiety"]],
                 "Gender",
                 ["Beck Anxiety", "Hamilton Anxiety"],
                 "Anxiety Level",
                 "markers",
                 ["diamond","cross"],
                 [63,56],
                 [["Minimum</br></br>(0-7)","Mild</br></br>(8-15)","Moderate</br></br>(16-25)","Severe</br></br>(26-63)"],
                  ["Minimum</br></br>(0-7)","Mild</br></br>(8-14)","Moderate</br></br>(15-23)","Severe</br></br>(24-56)"]])

push_viz_bar_subplots(2, 
             demographic_df.loc[:, ["Gender","Beck Anxiety", "Hamilton Anxiety"]],
             "Gender_bar",
             ["Beck Anxiety", "Hamilton Anxiety"],
             "Anxiety Level",
             [63,56],
                 [["Minimum</br></br>(0-7)","Mild</br></br>(8-15)","Moderate</br></br>(16-25)","Severe</br></br>(26-63)"],
                  ["Minimum</br></br>(0-7)","Mild</br></br>(8-14)","Moderate</br></br>(15-23)","Severe</br></br>(24-56)"]])
##############################################################################################




# Graph in order of BMI '

push_viz_scatter_subplots(2, 
                 demographic_df.loc[:, ["BMI","Beck Anxiety", "Hamilton Anxiety"]],
                 "BMI",
                 ["Beck Anxiety", "Hamilton Anxiety"],
                 "Anxiety Level",
                 "markers",
                 ["diamond","cross"],
                 [63,56],
                 [["Minimum</br></br>(0-7)","Mild</br></br>(8-15)","Moderate</br></br>(16-25)","Severe</br></br>(26-63)"],
                  ["Minimum</br></br>(0-7)","Mild</br></br>(8-14)","Moderate</br></br>(15-23)","Severe</br></br>(24-56)"]])
push_viz_bar_subplots(2, 
             demographic_df.loc[:, ["BMI","Beck Anxiety", "Hamilton Anxiety"]],
             "BMI_bar",
             ["Beck Anxiety", "Hamilton Anxiety"],
             "Anxiety Level",
             [63,56],
                 [["Minimum</br></br>(0-7)","Mild</br></br>(8-15)","Moderate</br></br>(16-25)","Severe</br></br>(26-63)"],
                  ["Minimum</br></br>(0-7)","Mild</br></br>(8-14)","Moderate</br></br>(15-23)","Severe</br></br>(24-56)"]])