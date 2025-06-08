#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 23:00:02 2022

@author: octopusphoenix
"""

import pandas as pd
import  numpy as np
from metadata_toolbox import get_bmi



#Loading  Metadata
#============================================================================================================


anxiety_df= pd.read_excel('Anxiety Scale.xlsx', sheet_name=None) 
demographic_df= pd.read_excel('Demographic data.xlsx') 




#Cleaning  Metadata
#============================================================================================================


demographic_df = demographic_df[demographic_df["Participant ID"] != "A17"]



#getting responses  for Beck and Hamilton inventory
df_beck= anxiety_df.get('Beck Anxiety Inventory').T
df_hamilton= anxiety_df.get('Hamilton Anxiety Scale')



#column list to sum for hamilton scale
col_list_hamilton= list(df_hamilton)
col_list_hamilton.remove("Unnamed: 0")



#calculating anxiety for the two scales
beck_anxiety =df_beck[1:].sum(axis=1).tolist()
hamilton_anxiety =df_hamilton[col_list_hamilton].sum(axis=1).tolist()  



#saving the calculated values in the dataframe
demographic_df["Beck Anxiety"]= beck_anxiety
demographic_df["Hamilton Anxiety"]= hamilton_anxiety


#calculating BMI  and saving in the dataframe
demographic_df['BMI'] = demographic_df.apply(lambda x: get_bmi(x["Height (cm)"], x["Weight (lbs)"]), axis=1)



bins = [0, 10, 20,30, 40,50,60, np.inf]
names = [ '<10', '10-20', '20-30','30-40', '40-50','50-60','60+']

demographic_df['AgeRange'] = pd.cut(demographic_df['Age'], bins, labels=names)

new_particpant_id= np.arange(1,20,1,dtype='int')
demographic_df['Participant ID']= new_particpant_id

demographic_df.to_csv('Metadata.csv')



