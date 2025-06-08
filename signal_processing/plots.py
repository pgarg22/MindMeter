#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 03:57:56 2023

@author: octopusphoenix
"""

import neurokit2 as nk
from Load_ecg_res_data import *
import matplotlib.pyplot as plt
import pandas as pd

# rsp=data_ecg_res[0][:15000,1]
# cleaned = nk.rsp_clean(data_ecg_res[0][:15000,1], sampling_rate=500)

# info = nk.rsp_findpeaks(cleaned)

# info = nk.rsp_fixpeaks(info)

# plt.rcParams.update({'font.size': 22})
# nk.events_plot([info["RSP_Peaks"], info["RSP_Troughs"]], cleaned)
# fig = plt.gcf()
# plt.xlabel("Samples")
# plt.ylabel("RSP (mV)")
# fig.set_size_inches(18.5, 10.5)


# peak_signal, info = nk.rsp_peaks(cleaned, sampling_rate=500)

# data = pd.concat([pd.DataFrame({"RSP": rsp}), peak_signal], axis=1)

# fig = nk.signal_plot(data)
# plt.ylabel("Voltage (mV)")




plt.rc('font', size=22)


ecg_cleaned = nk.ecg_clean(data_ecg_res[0][:,0], sampling_rate=500)

peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=500, correct_artifacts=True)

hrv_indices = nk.hrv(peaks, sampling_rate=500, show=True)
fig = plt.gcf()
allaxes = fig.get_axes()
fig.set_size_inches(18.5, 12.5)
plt.tight_layout()
fig.subplots_adjust(wspace=0.2)



plt.rc('font', size=25)
hrv = nk.hrv_time(peaks, sampling_rate=500, show=True)
fig = plt.gcf()
plt.ylabel("Count")
fig.set_size_inches(18.5, 9)




plt.rc('font', size=30)
hrv_welch = nk.hrv_frequency(peaks, sampling_rate=500, show=True)
fig = plt.gcf()
plt.legend(fontsize=30) 
fig.set_size_inches(18.5, 9)

# Clean signal
rsp= data_ecg_res[0][:,1]
cleaned = nk.rsp_clean(rsp, sampling_rate=500)

# Extract peaks
df, peaks_dict = nk.rsp_peaks(cleaned) 
info = nk.rsp_fixpeaks(peaks_dict)
formatted = nk.signal_formatpeaks(info, desired_length=len(cleaned),peak_indices=info["RSP_Peaks"])
candidate_peaks = nk.events_plot(peaks_dict['RSP_Peaks'], cleaned)
fixed_peaks = nk.events_plot(info['RSP_Peaks'], cleaned)
# Extract rate
rsp_rate = nk.rsp_rate(cleaned, peaks_dict, sampling_rate=500)

# Visualize
fig.set_size_inches(18.5, 12)
nk.signal_plot(rsp_rate, sampling_rate=500)
plt.ylabel('Breaths Per Minute')
plt.rc('font', size=40)

params = {'axes.labelsize': 30,
          'axes.titlesize': 30}
plt.rcParams.update(params)

rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=500, show=True)
fig = plt.gcf()
plt.legend(fontsize=30) 
plt.xlabel("BB_n (ms)",fontsize=30)
plt.ylabel("BB_n+1 (ms)",fontsize=30)
plt.title("Poincaré Plot",fontsize=30)
plt.rcParams.update({'font.size': 30})
fig.set_size_inches(18.5, 12)




