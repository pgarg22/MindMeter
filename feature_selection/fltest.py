#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:26:05 2023

@author: octopusphoenix
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest



from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


from functions_toolbox import transform_categorical


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from functions_toolbox import feature_selector_basic

from sklearn.ensemble import ExtraTreesClassifier

import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import mutual_info_classif

def cor_selector(X, y,num_feats):
    
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature





def rfe_logistic(X, y,num_feats):

    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X,y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return(rfe_support,rfe_feature)


def rfe_logistic_lbfgs(X, y,num_feats):

    rfe_selector = RFE(estimator=LogisticRegression(solver='lbfgs'), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X,y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return(rfe_support,rfe_feature)    


def embedded_lr_feature_selector(X, y,num_feats):
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embeded_lr_selector.fit(X, y)

    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    return(embeded_lr_support,embeded_lr_feature)

   

def embedded_rf_feature_selector(X, y,num_feats):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    return(embeded_rf_support,embeded_rf_feature)


def chi_selector(X, y,num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return(chi_support,chi_feature)






def Extra_tree_selector(X, y,num_feats):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.sort_values(ascending=False,inplace=True)
    selected_feature=feat_importances.keys().to_list()[0:num_feats]
    
    feature_name = X.columns.tolist()
    et_support = [True if i in selected_feature else False for i in feature_name]
    
    return(et_support,selected_feature)
    


    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(13).plot.bar()
    list1=feat_importances.keys().to_list()



def select_k_best_score_selector(X, y,num_feats):
    test = SelectKBest(score_func=f_classif, k=num_feats)
    fit = test.fit(X, y)
    kbest_support = test.get_support()
    kbest_feature = X.loc[:,kbest_support].columns.tolist()
    return(kbest_support,kbest_feature)


def forward_feature_selector(X, y,num_feats):
    
    feature_name = X.columns.tolist()
    
    forward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=num_feats,
           forward=True,
           verbose=2,
           scoring='roc_auc',
           cv=4,
           n_jobs=-1)
    fselector = forward_feature_selector.fit(X,y)
    features= fselector.k_feature_names_
    support = [True if i in features else False for i in feature_name]
    return support, features


def forward_feature_selector(X, y,num_feats):
    
    feature_name = X.columns.tolist()
    
    forward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=num_feats,
           forward=True,
           floating=False,
           verbose=2,
           scoring='roc_auc',
           cv=4,
           n_jobs=-1)
    fselector = forward_feature_selector.fit(X,y)
    features= fselector.k_feature_names_
    support = [True if i in features else False for i in feature_name]
    return support, features

def forward_feature_selector_floating(X, y,num_feats):
    
    feature_name = X.columns.tolist()
    
    forward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=num_feats,
           forward=True,
           floating=True,
           verbose=2,
           scoring='roc_auc',
           cv=4,
           n_jobs=-1)
    fselector = forward_feature_selector.fit(X,y)
    features= fselector.k_feature_names_
    support = [True if i in features else False for i in features]
    return support, features




def backward_feature_selector(X, y,num_feats):
    
    feature_name = X.columns.tolist()
    
    forward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=num_feats,
           forward=False,
           floating=False,
           verbose=2,
           scoring='roc_auc',
           cv=4,
           n_jobs=-1)
    fselector = forward_feature_selector.fit(X,y)
    features= fselector.k_feature_names_
    support = [True if i in features else False for i in feature_name]
    return support, features







def backward_feature_selector_floating(X, y,num_feats):
    
    feature_name = X.columns.tolist()
    
    forward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=num_feats,
           forward=False,
           floating=True,
           verbose=2,
           scoring='roc_auc',
           cv=4,
           n_jobs=-1)
    fselector = forward_feature_selector.fit(X,y)
    features= fselector.k_feature_names_
    support = [True if i in features else False for i in feature_name]
    return support, features






def mutual_info_selector(X, y,num_feats):
    
    # determine the mutual information
    mutual_info = mutual_info_classif(X, y)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X.columns
    mutual_info.sort_values(ascending=False)
    
    selected_feature=mutual_info.keys().to_list()[0:num_feats]
    
    feature_name = X.columns.tolist()
    mi_support = [True if i in selected_feature else False for i in feature_name]
    
    return(mi_support,selected_feature)

    


def mutual_info_selector2(X, y,num_feats):
    
    # determine the mutual information
    mutual_info = mutual_info_classif(X, y)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X.columns
    mutual_info.sort_values(ascending=False)
    
 
    sel_five_cols = SelectKBest(mutual_info_classif, k=num_feats)
    sel_five_cols.fit(X, y)
    
    mi_support= sel_five_cols.get_support()
    selected_feature=X.columns[sel_five_cols.get_support()]
    return(mi_support,selected_feature)
 


df_event_features= pd.read_csv("event_features_ecg_rsp.csv")

X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
y=df_event_features['Condition']  #



 
X_train=X
y_train=y

# , X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


#X_train= feature_selector_basic(X_train,0.9)

num_feats=10

X_train= feature_selector_basic(X_train,0.95)

y_train, encoder= transform_categorical(y_train)

   

cor_support, cor_feature = cor_selector(X_train, y_train,num_feats)
print(str(len(cor_feature)), 'selected features')

rfe_support, rfe_logistic_features= rfe_logistic(X_train, y_train,num_feats)
print(str(len(rfe_logistic_features)), 'selected features')



embeded_lr_support,embeded_lr_feature= embedded_lr_feature_selector(X_train, y_train,num_feats)
print(str(len(embeded_lr_feature)), 'selected features')


embeded_rf_support,embeded_rf_feature= embedded_rf_feature_selector(X_train, y_train,num_feats)
print(str(len(embeded_rf_feature)), 'selected features')

chi_support, chi_feature=  chi_selector(X_train, y_train,num_feats)
print(str(len(chi_feature)), 'selected features')
















feature_name=X_train.columns



# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


