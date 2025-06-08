#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:17:51 2023

@author: octopusphoenix
"""

from Load_ecg_res_data import data_ecg_res
import pandas as pd
from signal_processing_toolbox import run_ecg_res_processing

time_window=[10,20,30,40]
# names=["trans10","trans20","trans30"]

# for i in range(0,len(names)):
#     run_ecg_res_processing(data_ecg_res,time_window[i],time_window[i],
#                            names[i],drop_missing=False)
    


just_start_names=["trans10_onlybegin","trans20_onlybegin"
                  ,"trans30_onlybegin","trans40_onlybegin"]



for i in range(0,len(just_start_names)):
    run_ecg_res_processing(data_ecg_res,time_window[i],0,
                           just_start_names[i],drop_missing=False)
