#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:00:16 2023

@author: octopusphoenix
"""

from Load_ecg_res_data import data_ecg_res
import pandas as pd
from signal_processing_random_labels import run_ecg_res_processing
import random

time_window=[10,20,30,40]
names=["trans10","trans20","trans30"]


condition_list=["1","2","3","4","5","6","7","8"]
condition_list2= ["Anxious", "Normal", "Anxious", "Normal","Anxious", "Normal","Anxious", "Normal"]

random.shuffle(condition_list2)


run_ecg_res_processing(data_ecg_res,condition_list,0,0,
                        "trans0","video_number",drop_missing=False)


run_ecg_res_processing(data_ecg_res,condition_list2,0,0,
                        "trans0","random_anxiety_label",drop_missing=False)


for i in range(0,len(names)):
    run_ecg_res_processing(data_ecg_res,condition_list,time_window[i],time_window[i],
                            names[i],"video_number",drop_missing=False)
    
    run_ecg_res_processing(data_ecg_res,condition_list2,time_window[i],time_window[i],
                            names[i],"random_anxiety_label",drop_missing=False)
    



just_start_names=["trans10_onlybegin","trans20_onlybegin"
                  ,"trans30_onlybegin","trans40_onlybegin"]





for i in range(0,len(just_start_names)):
    run_ecg_res_processing(data_ecg_res,condition_list,time_window[i],0,
                           just_start_names[i],"video_number",drop_missing=False)
    
    
    run_ecg_res_processing(data_ecg_res,condition_list2,time_window[i],0,
                           just_start_names[i],"random_anxiety_label",drop_missing=False)