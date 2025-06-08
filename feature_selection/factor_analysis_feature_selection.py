#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:27:59 2023

@author: octopusphoenix
"""


# Import required libraries
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


    
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

import numpy as np


# def slope(y2,y1,x2, x1):
#     m = (y2-y1)/(x2-x1)
#     return m

df_event_features= pd.read_csv("event_features_ecg_rsp.csv")

X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y=df_event_features['Condition']  #

sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)



#CHECK ADEQUACY
#Bartlett
#p-value should be 0 (statistically sig.)
chi_square_value,p_value=calculate_bartlett_sphericity(X_std)
print(chi_square_value, p_value)

#KMO
#Value should be 0.6<
kmo_all,kmo_model=calculate_kmo(X_std)
print(kmo_model)




#EXPLORATORY FACTOR ANALYSIS
fa = FactorAnalyzer(25, rotation=None)
fa.fit(X_std)

est= fa._get_factor_variance(fa.loadings_)

ev,vec= fa.get_eigenvalues()
xtic= np.arange(1, X_std.shape[1]+1, 1, dtype=int)
xi = list(range(len(xtic)))
range(1,X_std.shape[1]+1)
# SCREEPLOT (need pyplot)

figure(figsize=(30, 20), dpi=80)
plt.scatter(range(1,X_std.shape[1]+1),ev)
plt.plot(range(1,X_std.shape[1]+1),ev)
plt.xticks(xi,xtic )
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid(visible= 'True',which='both', axis='x')
plt.show()


# slope_value=[]
# for i in range(len(ev)-1):
#     i += 1
#     v=slope(ev[i], ev[i-1], i, (i-1))
#     print(i,v)
#     slope_value.append(v)
    


fa = FactorAnalyzer(19, rotation="varimax", method='minres', use_smc=True)
fa.fit(X_std)



loadings = pd.DataFrame(fa.loadings_, 
                        columns=['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4',
                                 'Factor 5', 'Factor 6', 'Factor 7', 'Factor 8',
                                 'Factor 9', 'Factor 10', 'Factor 11', 'Factor 12',
                                 'Factor 13', 'Factor 14', 'Factor 15', 'Factor 16',
                                 'Factor 17', 'Factor 18', 'Factor 19'
                                ], index=X.columns)
print('Factor Loadings \n%s' %loadings)

commonalities= fa.get_communalities()
test= fa._get_factor_variance(fa.loadings_)


loadings_abs= loadings.abs()
feat_load= loadings_abs.idxmax().to_list()
features_list= list(dict.fromkeys(feat_load))


    
# loadings_total= df2 = loadings_abs.sum(axis=1)




# loadings_total.sort_values(ascending=False, inplace= True)


"HRV_"


