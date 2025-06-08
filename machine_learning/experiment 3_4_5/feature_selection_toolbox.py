#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:02:21 2023

@author: octopusphoenix
"""


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA


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




def rfe_lr_selectory(X, y,num_feats,penalty_type):
    rfe_selector = RFE(estimator=LogisticRegression(penalty=penalty_type), n_features_to_select=num_feats, step=1, verbose=1)
    rfe_selector.fit(X,y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return(rfe_support,rfe_feature)    


def rfe_random_forest(X, y,num_feats):
    rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=500,random_state=0), n_features_to_select=num_feats, step=1, verbose=1)
    rfe_selector.fit(X,y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return(rfe_support,rfe_feature)    




def Extra_tree_selector(X, y,num_feats):
    model = ExtraTreesClassifier(n_estimators = 500, criterion ='gini',random_state=0)
    model.fit(X, y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.sort_values(ascending=False,inplace=True)
    selected_feature=feat_importances.keys().to_list()[0:num_feats]
    
    feature_name = X.columns.tolist()
    et_support = [True if i in selected_feature else False for i in feature_name]
    
    return(et_support,selected_feature)
    


def select_k_best_score_selector(X, y,num_feats,scoring_func):
    X_norm= MinMaxScaler().fit_transform(X)
    test = SelectKBest(score_func=scoring_func, k=num_feats)
    test.fit(X_norm, y)
    kbest_support = test.get_support()
    kbest_feature = X.loc[:,kbest_support].columns.tolist()
    return(kbest_support,kbest_feature)




def forward_feature_selector(X, y,num_feats,floating_bool):
    
    feature_name = X.columns.tolist()
    
    forward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators=500,random_state=0,n_jobs=-1),
           k_features=num_feats,
           forward=True,
           floating=floating_bool,
           verbose=1,
           scoring='roc_auc',
           cv=4,
           n_jobs=-1)
    fselector = forward_feature_selector.fit(X,y)
    features= fselector.k_feature_names_
    support = [True if i in features else False for i in feature_name]
    return support, features



def backward_feature_selector(X, y,num_feats,floating_bool):
    
    feature_name = X.columns.tolist()
    
    forward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators=500,random_state=0,n_jobs=-1),
           k_features=num_feats,
           forward=False,
           floating=floating_bool,
           verbose=1,
           scoring='roc_auc',
           cv=4,
           n_jobs=-1)
    fselector = forward_feature_selector.fit(X,y)
    features= fselector.k_feature_names_
    support = [True if i in features else False for i in feature_name]
    return support, features

def factor_analysis_selector(X, y,num_feats):
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    
    fa = FactorAnalyzer((2*num_feats), rotation="varimax", method='minres', use_smc=True)
    fa.fit(X_std)
    
    columns=[]
    for i in range(1,((2*num_feats)+1)):
        columns.append("Factor "+ str(i))
    
    loadings = pd.DataFrame(fa.loadings_, 
                            columns=columns, index=X.columns)
    loadings_abs= loadings.abs()
    feat_load= loadings_abs.idxmax().to_list()
    features_list= list(dict.fromkeys(feat_load))
    selected_feature= features_list[0:num_feats]
    feature_name = X.columns.tolist()
    fa_support = [True if i in selected_feature else False for i in feature_name]
    
    return(fa_support,selected_feature)


def pca_selector(X, y,num_feats):
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    pca = PCA(n_components=(2*num_feats))
    fit = pca.fit(X_std)
    # summarize components
    list1= fit.components_
    list2= list1.transpose()

    columns=[]
    for i in range(1,((2*num_feats)+1)):
        columns.append("Component "+ str(i))

    df_components = pd.DataFrame(list2, index=X.columns,columns=columns)
    df_components_abs= df_components.abs()
    feat_pca= df_components_abs.idxmax().to_list()
    features_list= list(dict.fromkeys(feat_pca))
    selected_feature= features_list[0:num_feats]
    feature_name = X.columns.tolist()
    pca_support = [True if i in selected_feature else False for i in feature_name]
    return(pca_support,selected_feature)

def run_all_selectors(X, y,num_feats):
    
    cor_support, cor_feature = cor_selector(X, y,num_feats)
    lr_l2_support, lr_l2_features= rfe_lr_selectory(X, y,num_feats,penalty_type='l2')
    rf_support,rf_feature= rfe_random_forest(X, y,num_feats)
    ets_support, ets_feature = Extra_tree_selector(X, y,num_feats)
    
    chi_support, chi_feature=  select_k_best_score_selector(X, y,num_feats,scoring_func=chi2)
    
    
    ffs_support,ffs_feature= forward_feature_selector(X, y,num_feats, floating_bool='False')
    bfs_support, bfs_feature=  backward_feature_selector(X, y,num_feats, floating_bool='False')
    
    
    mis_support, mis_features= select_k_best_score_selector(X, y,num_feats,mutual_info_classif)
    
    
    fa_support, fa_feature = factor_analysis_selector(X, y,num_feats)
    pca_support, pca_feature = pca_selector(X, y,num_feats)
    
    
    # put all selection together
    feature_name=X.columns

    feature_selection_df = pd.DataFrame({'Feature':feature_name, 
                                         'Pearson Coefficent Selector':cor_support, 
                                         'Logistic Regression Selector':lr_l2_support,
                                         'Random Forest Selector':rf_support,
                                         'Extra Tree Selector':ets_support,
                                         'Chi-2 Selector':chi_support, 
                                          'Forward Feature Selector':ffs_support,
                                          'Backward Feature Selector':bfs_support,
                                         'Mutual Info Selector':mis_support,
                                         'Factor Analysis Selector':fa_support,
                                         'PCA selector': pca_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    
    
    features_list= []
    features_list.append(cor_feature)
    features_list.append(lr_l2_features)
    features_list.append(rf_feature)
    features_list.append(ets_feature)
    features_list.append(chi_feature)
    features_list.append(list(ffs_feature))
    features_list.append(list(bfs_feature))
    features_list.append(mis_features)
    features_list.append(fa_feature)
    features_list.append(pca_feature)
    
    selectors_list=['Pearson Coefficient Feature Selector',
                    'Logistic Regression Feature Selector',
                    'Random Forest Feature Selector',
                    'Extra Tree Feature Selector',
                    'Chi-2 Feature Selector',
                    'Sequential Forward Feature Selector',
                    'Sequential Backward Feature Selector',
                    'Mutual Info Feature Selector',
                    'Factor Analysis Selector',
                    'PCA selector']
    
    return(feature_selection_df,features_list,selectors_list)





 