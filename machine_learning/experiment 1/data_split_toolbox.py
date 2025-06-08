#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:02:59 2023

@author: octopusphoenix
"""



from machine_learning_toolbox import  transform_categorical, scale_numerical_standard 
from machine_learning_toolbox import run_random_forest_without_fs, run_xgb_without_fs
from machine_learning_toolbox import run_svm_without_fs
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
from graphs_ml_results_toolbox import push_ml_results_trans



def participant_train_test_split_without_fs(participant_id,df_event_features,test_name,directory):
    train= df_event_features[df_event_features['Participant'] != participant_id]
    test=  df_event_features[df_event_features['Participant'] == participant_id]
    
    X_train =train.loc[:, ~train.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    X_test =test.loc[:, ~test.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    y_train=train['Condition']  
    y_test=test['Condition']  



    X_train, train_scaler=scale_numerical_standard(X_train)
    X_test= scale_numerical_standard(X_test, train_scaler)
    y_train, encoder= transform_categorical(y_train)
    y_test=transform_categorical(y_test,encoder)


    
    print("Paricipant :" + str(participant_id))
    
    rf_score= run_random_forest_without_fs (X_train, X_test, y_train, y_test,test_name+"_Participant_"+str(participant_id),directory )
    xgb_score= run_xgb_without_fs(X_train, X_test, y_train, y_test,test_name+"_Participant_"+str(participant_id),directory)
    svm_score= run_svm_without_fs(X_train, X_test, y_train, y_test,test_name+"_Participant_"+str(participant_id),directory)


    return(rf_score+ xgb_score +svm_score)



def video_train_test_split_without_fs(label_id,df_event_features,test_name,directory):
    train= df_event_features[df_event_features['Label'] != label_id]
    test=  df_event_features[df_event_features['Label'] == label_id]
    
    X_train =train.loc[:, ~train.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    X_test =test.loc[:, ~test.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    y_train=train['Condition']  
    y_test=test['Condition']  



    X_train, train_scaler=scale_numerical_standard(X_train)
    X_test= scale_numerical_standard(X_test, train_scaler)
    y_train, encoder= transform_categorical(y_train)
    y_test=transform_categorical(y_test,encoder)


    
    print("Video :" + str(label_id))
    
    rf_score= run_random_forest_without_fs (X_train, X_test, y_train, y_test,test_name+"_Video_"+str(label_id),directory)
    xgb_score= run_xgb_without_fs(X_train, X_test, y_train, y_test,test_name+"_Video_"+str(label_id),directory)
    svm_score= run_svm_without_fs(X_train, X_test, y_train, y_test,test_name+"_Video_"+str(label_id),directory)


    return(rf_score+ xgb_score +svm_score)





def all_video_transitional_without_fs(df_event_features,test_name,directory):
    df_video = pd.DataFrame(columns=['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score',  
                                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score',
                                'svm_accuracy','svm_precision', 'svm_recall', 'svm_f1score'])
    
    for i in range(1,9):
        
        rslt= video_train_test_split_without_fs(i, df_event_features,test_name,directory)
        df_video.loc[len(df_video)] = rslt
        
        
    df_video.to_csv(directory+test_name+'_Video_split_results.csv')
 
    
 
    
 
def all_participant_transitional_without_fs(df_event_features,test_name,directory):

    df_participant = pd.DataFrame(columns=['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score',  
                                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score',
                                'svm_accuracy','svm_precision', 'svm_recall', 'svm_f1score'])

    for i in range(1,20):
        
        rslt= participant_train_test_split_without_fs(i, df_event_features,test_name,directory)
        df_participant.loc[len(df_participant)] = rslt
        
        
    df_participant.to_csv(directory+test_name+'_Participant_split_results.csv')





def random_split_test_without_fs(df_event_features,test_name,directory):

    X=df_event_features.loc[:, ~df_event_features.columns.isin(['Condition','Label','Unnamed: 0','Participant'])]
    y=df_event_features['Condition']  #
    df_random = pd.DataFrame(columns=['rf_accuracy','rf_precision', 'rf_recall', 'rf_f1score',  
                                'xgb_accuracy', 'xgb_precision', 'xgb_recall', 'xgb_f1score',
                                'svm_accuracy','svm_precision', 'svm_recall', 'svm_f1score'])
    
    
    skf = StratifiedKFold(n_splits=6, random_state=42,shuffle=True)
    skf.get_n_splits(X, y)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        X_train, train_scaler=scale_numerical_standard(X_train)
        X_test= scale_numerical_standard(X_test, train_scaler)
        y_train, encoder= transform_categorical(y_train)
        y_test=transform_categorical(y_test,encoder)
        
        rf_score=run_random_forest_without_fs (X_train, X_test, y_train, y_test,test_name+"_Random_split",directory)
        xgb_score= run_xgb_without_fs(X_train, X_test, y_train, y_test,test_name+"_Random_split",directory)
        svm_score=run_svm_without_fs(X_train, X_test, y_train, y_test,test_name+"_Random_split",directory)
        
        rslt= rf_score+ xgb_score +svm_score
        df_random.loc[len(df_random)] = rslt
    
    df_random.to_csv(directory+test_name+'_random_split_results.csv')


    

        
def run_multiple_tests_without_fs_trans(df,cols_list,test_name_list):
    if len(cols_list) != len(test_name_list):
        raise Exception("Column list length doesnt match test names list")
    
    if len(cols_list)<1:
        raise Exception("Column list has no feature set")
    
    
    if not os.path.exists("test_results"):
        os.mkdir("test_results")

    res_directory = os.getcwd() + "/test_results/"
    for i in range(0,len(cols_list)):
        
        cols= cols_list[i]
        test_name= test_name_list[i]
        df_filtered= df[cols]
        
        if not os.path.exists(res_directory+test_name):
            os.mkdir(res_directory+test_name)
        directory = res_directory+test_name+"/" 
        
        file = open(directory+'features_used_with_'+test_name+'.txt','w')
        file.write("Features Used with for ")
        file.write(test_name+"\n\n")
        for c in range(0,len(cols)-4):
            item= cols[c]
            file.write(str(c+1)+". "+item+ "\n")
        file.close()
        
        random_split_test_without_fs(df_filtered,test_name,directory)
        all_participant_transitional_without_fs(df_filtered, test_name,directory)
        all_video_transitional_without_fs(df_filtered, test_name,directory)
        push_ml_results_trans(directory,test_name)
    
    
   