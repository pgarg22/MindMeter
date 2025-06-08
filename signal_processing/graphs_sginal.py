#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:12:09 2023

@author: octopusphoenix
"""

from Load_ecg_res_data import data_ecg_res
import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import scipy

def percentage_list(my_list):
    percent_list= [ 10*elem for elem in my_list]
    return(percent_list)
# def signaltonoise(a, axis=0, ddof=0):
#     a = np.asanyarray(a)
#     m = a.mean(axis)
#     sd = a.std(axis=axis, ddof=ddof)
#     return np.where(sd == 0, 0, m/sd)
# signal_freq_band = [2, 40]  

snr_ecg=[]
snr_rsp=[]
# for k in range(0,len(data_ecg_res)):
#     # snr_ecg.append(signaltonoise(data_ecg_res[k][:,0]))
#     # snr_rsp.append(signaltonoise(data_ecg_res[k][:,1]))
    
#     # f, pxx_den = scipy.signal.periodogram((data_ecg_res[k][:,0]), fs=500, scaling="spectrum")
#     # if sum(pxx_den):
#     #     signal_power = sum(pxx_den[(signal_freq_band[0]*10):(signal_freq_band[1]*10)])
#     #     SNR = signal_power / (sum(pxx_den) - signal_power)

#     # snr_ecg.append(SNR)
    
    # data_clean, info = nk.bio_process(ecg=data_ecg_res[k][:,0], sampling_rate=500)
    # noise= np.subtract(data_ecg_res[k][:,0], data_clean) 
    # snr_all= np.divide(data_clean, noise)
    
    # snr_ecg.append(np.mean(snr_all))
    
    
#     # data_clean, info = nk.bio_process(ecg=data_ecg_res[k][:,1], sampling_rate=500)
#     # noise= np.subtract(data_ecg_res[k][:,1], data_clean) 
    
#     # snr_all= np.divide(data_clean, noise)
    
#     # snr_rsp.append(np.mean(snr_all))
    
# print("Min SNR ECG", np.min(snr_ecg))
# print("Max SNR ECG", np.max(snr_ecg))
# # print("Min SNR RSP", np.min(snr_rsp))
# # print("Max SNR RSP", np.max(snr_rsp))


import plotly.graph_objects as go

df_metadata=pd.read_csv("Metadata.csv")
 

bins = [-1, 7, 15,25, 63]
bins2 = [-1, 7, 14,23, 56]
names = [ 'Minimum', 'Mild', 'Moderate','Severe']

df_metadata['Beck_class'] = pd.cut(df_metadata['Beck Anxiety'], bins, labels=names)

df_metadata['Hamilton_class'] = pd.cut(df_metadata['Hamilton Anxiety'], bins2, labels=names)


color_dict = {'Minimum': "lightseagreen",'Mild': "yellow",'Moderate': "indianred",'Severe': "blue"}


names = [ 'lightseagreen', 'yellow', 'indianred','blue']
df_metadata['Beck_color'] = pd.cut(df_metadata['Beck Anxiety'], bins, labels=names)

df_metadata['Hamilton_color'] = pd.cut(df_metadata['Hamilton Anxiety'], bins2, labels=names)


values = df_metadata['Beck_color'].to_list()
keys2 = [1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
keys= [str(i) for i in keys2]
dictionary_beck = dict(zip(keys, values))

values_ham=  df_metadata['Hamilton_color'].to_list()
dictionary_ham = dict(zip(keys, values_ham))
fig2 = go.Figure()




# for k in range(0,len(data_ecg_res)):
  
#     fig2.add_trace(go.Box( name= str(k+1),
#         y=data_ecg_res[k][:,1],
#         marker_color = color_dict.get(df_metadata['Beck_class'][k]),
         
#         ))


# fig2.update_layout(
#     title="Box plot of respiration signal data for all participants",
#                       xaxis_title= "Participants",
#                       yaxis_title="Respiration(mV)",
#     yaxis=dict(
#         autorange=True,
#         showgrid=True,
#         zeroline=True,
#         dtick=5,
#         gridcolor='rgb(255, 255, 255)',
#         gridwidth=1,
#         zerolinecolor='rgb(255, 255, 255)',
#         zerolinewidth=2,
#     ),
#     margin=dict(
#         l=40,
#         r=30,
#         b=80,
#         t=100,
#     ),
#     paper_bgcolor='rgb(243, 243, 243)',
#     plot_bgcolor='rgb(243, 243, 243)',
#     showlegend=False
# )


# if not os.path.exists("images"):
#     os.mkdir("images")
# fig2.write_image("images/"+"RSP_Amplitude_beck"+".png")




# fig = go.Figure()




# for k in range(0,len(data_ecg_res)):
  
#     fig.add_trace(go.Box( name= str(k+1),
#         y=data_ecg_res[k][:,1],
#         marker_color = color_dict.get(df_metadata['Hamilton_class'][k]),
         
#         ))


# fig.update_layout(
#     title="Box plot of respiration signal data for all participants",
#                       xaxis_title= "Participants",
#                       yaxis_title="Respiration(mV)",
#     yaxis=dict(
#         autorange=True,
#         showgrid=True,
#         zeroline=True,
#         dtick=5,
#         gridcolor='rgb(255, 255, 255)',
#         gridwidth=1,
#         zerolinecolor='rgb(255, 255, 255)',
#         zerolinewidth=2,
#     ),
#     margin=dict(
#         l=40,
#         r=30,
#         b=80,
#         t=100,
#     ),
#     paper_bgcolor='rgb(243, 243, 243)',
#     plot_bgcolor='rgb(243, 243, 243)',
#     showlegend=False
# )


# if not os.path.exists("images"):
#     os.mkdir("images")
# fig.write_image("images/"+"RSP_Amplitude_hamilton"+".png")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

wid= [0.2 for i in range(0,19)]

# data2 = pd.DataFrame({"1": data_ecg_res[0][:,0], })
df=[]
for i in range(0,len(data_ecg_res)):
    df2=pd.DataFrame({"ECG": data_ecg_res[i][:,0], })
    df2["Participant"]= str(i+1)
    df.append(df2)
data2= pd.concat(df)



fig, bp = plt.subplots(figsize=(16,9))
sns.set(style='whitegrid')
sns.boxplot(
    data=data2,
    x="Participant", y="ECG",
    palette=dictionary_beck,width=0.7, dodge=False,
    
)
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
for line in bp.get_lines()[0::19]:
    line.set_color('yellow')
    
    
    
    
    
data = [x[:,0] for x in data_ecg_res]
    
    

fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
 
# Creating axes instance
bp = ax.boxplot(data, patch_artist = True,
                notch ='True',sym='')
 
colors = df_metadata['Beck_color'].to_list()
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# # changing color and linewidth of
# # whiskers
# for whisker, color in zip(bp['whiskers'], colors):
#     whisker.set(color =color,
#                 linewidth = 1.5,
#                 linestyle =":")
 
# # changing color and linewidth of
# # caps

# for cap, color in zip(bp['caps'], colors):
#     cap.set(color =color,
#             linewidth = 2)
 
# # changing color and linewidth of
# # medians
# for median, color in zip(bp['medians'], colors):
#     median.set(color =color,
#                linewidth = 3)
 
# # changing style of fliers
# for flier, color in zip(bp['fliers'], colors):
#     flier.set(marker ='D',
#               color =color,
              # alpha = 0.5)
    
plt.title("Box plot of ECG signal data for all participants")
plt.xlabel('Participants')
plt.ylabel('ECG(mv)')

plt.grid(False)
# x-axis labels    
    
    
# ax = plt.boxplot(data2)

# plt.show()

# plt.figure(figsize=(40,30))
# # option 1, specify props dictionaries
# c = "red"
# plt.boxplot(data2, notch=True, patch_artist=True,
#             boxprops=dict(facecolor=c, color=c),
#             capprops=dict(color=c),
#             whiskerprops=dict(color=c),
#             flierprops=dict(color=c, markeredgecolor=c),
#             medianprops=dict(color=c),
#             )
# plt.show()


# plt.xlim(0.5,4)
# plt.xticks([1,2,3], [1,2,3])
# plt.show()

# fig3 = go.Figure()


# min_ecg=0;
# max_ecg=0

# for k in range(0,len(data_ecg_res)):
#     data= percentage_list(data_ecg_res[k][:,0])
#     min_k=np.min(data)
#     max_k= np.min(data)
#     if(min_ecg>min_k):
#         min_ecg=min_k
#     if(max_ecg<max_k):
#         max_ecg=max_k
    
#     fig3.add_trace(go.Box( name= str(k+1),
#         y=data,
#         marker_color = color_dict.get(df_metadata['Beck_class'][k]),
         
#         ))


# fig3.update_layout(
#     title="Box plot of respiration signal data for all participants",
#                       xaxis_title= "Participants",
#                       yaxis_title="ECG(mV)",
#     yaxis=dict(
#         autorange=True,
#         showgrid=True,
#         zeroline=True,
#         dtick=5,
#         gridcolor='rgb(255, 255, 255)',
#         gridwidth=1,
#         zerolinecolor='rgb(255, 255, 255)',
#         zerolinewidth=2,
#     ),
#     margin=dict(
#         l=40,
#         r=30,
#         b=80,
#         t=100,
#     ),
#     paper_bgcolor='rgb(243, 243, 243)',
#     plot_bgcolor='rgb(243, 243, 243)',
#     showlegend=False
# )
# fig3.update_layout( boxgap=0.2)


# if not os.path.exists("images"):
#     os.mkdir("images")

# print("test")
# fig3.write_image("images/"+"ECG_Amplitude_beck.png")



# import plotly.graph_objects as go
# import numpy as np

# N = 19    # Number of boxes

# # generate an array of rainbow colors by fixing the saturation and lightness of the HSL
# # representation of colour and marching around the hue.
# # Plotly accepts any CSS color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.
# c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

# # Each box is represented by a dict that contains the data, the type, and the colour.
# # Use list comprehension to describe N boxes, each with a different colour and with different randomly generated data:
# fig = go.Figure(data=[go.Box(
#     y=data_ecg_res[i][:,0],
#     marker_color=c[i]
#     ) for i in range(int(N))])

# # format the layout
# fig.update_layout(
#     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#     yaxis=dict(zeroline=False, gridcolor='white'),
#     paper_bgcolor='rgb(233,233,233)',
#     plot_bgcolor='rgb(233,233,233)',
# )

# fig.show()
# fig.write_image("images/"+"ECG_Amplitude_beck.png")

# # for k in range(0,len(data_ecg_res)):
  
# #     fig2.add_trace(go.Box( name= "Participant "+str(k+1),
# #         y=data_ecg_res[k][:,0],
# #         marker_color = color_dict.get(df_metadata['Hamilton_class'][k]),
         
# #         ))


# # fig2.update_layout(
# #     title="Box plot of respiration signal data for all participants",
# #                       xaxis_title= "Participants",
# #                       yaxis_title="ECG(mV)",
# #     yaxis=dict(
# #         autorange=True,
# #         showgrid=True,
# #         zeroline=True,
# #         dtick=5,
# #         gridcolor='rgb(255, 255, 255)',
# #         gridwidth=1,
# #         zerolinecolor='rgb(255, 255, 255)',
# #         zerolinewidth=2,
# #     ),
# #     margin=dict(
# #         l=40,
# #         r=30,
# #         b=80,
# #         t=100,
# #     ),
# #     paper_bgcolor='rgb(243, 243, 243)',
# #     plot_bgcolor='rgb(243, 243, 243)',
# #     showlegend=False
# # )


# # if not os.path.exists("images"):
# #     os.mkdir("images")
# # fig2.write_image("images/"+"ECG_Amplitude_hamilton"+".png")


# # fig = go.Figure()
# # fig.add_trace(go.Box(
# #     y=[2.37, 2.16, 4.82, 1.73, 1.04, 0.23, 1.32, 2.91, 0.11, 4.51, 0.51, 3.75, 1.35, 2.98, 4.50, 0.18, 4.66, 1.30, 2.06, 1.19],
# #     name='Only Mean',
# #     marker_color='darkblue',
# #     boxmean=True # represent mean
# # ))
# # fig.add_trace(go.Box(
# #     y=[2.37, 2.16, 4.82, 1.73, 1.04, 0.23, 1.32, 2.91, 0.11, 4.51, 0.51, 3.75, 1.35, 2.98, 4.50, 0.18, 4.66, 1.30, 2.06, 1.19],
# #     name='Mean & SD',
# #     marker_color='royalblue',
# #     boxmean='sd' # represent mean and standard deviation
# # ))

# # fig.show()