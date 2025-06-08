#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:19:29 2022


Script to load ECG and respiration data of participants

@author: octopusphoenix
"""



"""
Loading  libraries and   updating plot styles
============================================================================================================
"""


from scipy.io import loadmat
import os 




"""
Loading  data
============================================================================================================
"""

mats = []
data_ecg_res=[]


dataDir = os.getcwd()
dataDir=dataDir +"/ecg_res_data/"

mats = []
for file in os.listdir( dataDir ) :
   annots= loadmat(dataDir+file )
   mats.append(annots)
   data=  annots.get("data")
   data_ecg_res.append(data)
   

"""
============================================================================================================
"""




