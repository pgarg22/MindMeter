#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 12:44:48 2023

@author: octopusphoenix
"""

from result_graph_toolbox import plot_std_bar_graph,plot_std_bar_graph_subplots, plot_std_bar_graph_subplots_trans_period

selector_names=["Best All selectors"]
num_feats_list=[5,10]

trans_names=["trans0", "trans10","trans20","trans30"
         ,"trans10_onlybegin","trans20_onlybegin",
        "trans30_onlybegin","trans40_onlybegin"]

selectors_list=['Pearson Coefficient Feature Selector',
                'Logistic Regression Feature Selector',
                'Random Forest Feature Selector',
                'Extra Tree Feature Selector',
                'Chi-2 Feature Selector',
                'Sequential Forward Feature Selector',
                'Sequential Backward Feature Selector',
                'Mutual Info Feature Selector',
                'Factor Analysis Selector',
                'PCA selector',
                'Best All Selectors']


metric_list=['accuracy','recall','precision','f1score']


for num_feats in num_feats_list:
    
    for metric in metric_list:
        plot_std_bar_graph_subplots_trans_period(selectors_list,trans_names, num_feats,"Participant",metric)
        plot_std_bar_graph_subplots_trans_period(selectors_list,trans_names, num_feats,"random",metric)
    plot_std_bar_graph_subplots_trans_period(selectors_list,trans_names, num_feats,"Video","accuracy")
    
    
    for tr_period in trans_names:
        for metric in metric_list:
            plot_std_bar_graph(selectors_list,tr_period,num_feats,"Participant",metric)
            plot_std_bar_graph(selectors_list,tr_period,num_feats,"random",metric)
            
        plot_std_bar_graph(selectors_list,tr_period,num_feats,"Video","accuracy")
        plot_std_bar_graph_subplots(selectors_list,tr_period,num_feats,metric_list,"Participant")
        plot_std_bar_graph_subplots(selectors_list,tr_period,num_feats,metric_list,"random")
        
# layout = go.Layout(
#     autosize=False,
#     width=2000,
#     height=1000,
# )

# for num_feats in num_feats_list:
#     for tr_period in trans_names:
#         rf_mean_list=[]
#         rf_std_list=[]
#         xgb_mean_list=[]
#         xgb__std_list=[]
#         svm_mean_list=[]
#         svm_std_list=[]
#         for selector in selectors_list:
#             test_name= selector+'_'+tr_period+'_features'+str(num_feats)
#             result_video= pd.read_csv("test_results/"+test_name+"/" +test_name+'_Video_split_results.csv')
#             result_video.drop("Unnamed: 0",axis=1,inplace=True)
            
#             result_video_mean=result_video.mean()
#             result_video_std=result_video.std()
            
#             rf_mean_list.append(result_video_mean['rf_accuracy'])
#             xgb_mean_list.append(result_video_mean['xgb_accuracy'])
#             svm_mean_list.append(result_video_mean['svc_accuracy'])
            
#             rf_std_list.append(result_video_std['rf_accuracy'])
#             xgb__std_list.append(result_video_std['xgb_accuracy'])
#             svm_std_list.append(result_video_std['svc_accuracy'])
        
        
#         fig = go.Figure(layout=layout)

#         fig.add_trace(go.Bar(
#                 name="Random Forest Accuracy",
#                 x=selectors_list, y=rf_mean_list,
#                 error_y=dict(type='data', array=rf_std_list)
#                 ))
        
#         fig.add_trace(go.Bar(
#                 name="XGB Accuracy",
#                 x=selectors_list, y=xgb_mean_list,
#                 error_y=dict(type='data', array=xgb__std_list)
#                 ))
        
#         fig.add_trace(go.Bar(
#                 name="SVM Accuracy",
#                 x=selectors_list, y=svm_mean_list,
#                 error_y=dict(type='data', array=svm_std_list)
#                 ))
#         fig.update_layout(barmode='group')
#         fig.show()

#         if not os.path.exists("Model_Trends"):
#               os.mkdir("Model_Trends")
#         fig.write_image("Model_Trends/Video"+str(num_feats)+"features_"+tr_period+"_accuracy"+".png")

        
# for num_feats in num_feats_list:
#     for tr_period in trans_names:
        
