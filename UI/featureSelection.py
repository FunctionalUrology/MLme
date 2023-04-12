#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:41:16 2021

@author: akshay
"""


import dash_bootstrap_components as dbc
from dash import dcc,html
from UI.scaling import genrateInfoCallback, genrateCollapseCallback,genrateAlertCallback,infoText
from UI.dataSampling import get_neighbors_Para
from app import app
from dash.dependencies import Input, Output,State


#get the header of each card of sampling tab
def getAlgoHeader(algo):
    return dbc.Row([
     dbc.Col(dbc.Checklist(options=[{"label": algo,"value": True}],
                                    value=[],
                                    id=algo,
                                    inline=True,switch=True,labelStyle={"font-weight": "bold",
                                                                        "font-size": "18px"},
                                    labelCheckedStyle={"color": "green"},persistence=True,persistence_type="memory"),
              width={"size": 9},),
   dbc.Col(dbc.Button("Parameters",id=algo+"-collapse-button",
            className="mr-1",size="sm",color="light",n_clicks=0,style={"margin-top": "15px"}),
        width={"size": 3},)
       ])

def getScoreFun(method):
    component= dbc.FormGroup([ 
                            dbc.Label("Function to calculate feature score",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "f_classif : ANOVA F-value between label/feature for classification tasks.", "value": "f_classif"},
                                    {"label": "chi2 : Chi-squared stats of non-negative features for classification tasks.", "value":"chi2"},
                                ],value= "f_classif", clearable=False,style={"font-size": "14px",'color': 'black'},
                                id=method+"-score_func",persistence=True,persistence_type="memory"),
                            ])
    return component


                            
                            
def getEstimator(method):
    component= dbc.FormGroup([ 
                            dbc.Label("Estimator",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "SVM", "value": "SVM"},
                                    {"label": "LogisticRegression", "value": "LogisticRegression"},
                                    {"label": "ExtraTrees", "value": "ExtraTrees"},                                    
                                    {"label": "DecisionTree", "value": "DecisionTree"},
                                    {"label": "LinearDiscriminantAnalysis", "value": "LinearDiscriminantAnalysis"},
                                ],value= "SVM", clearable=False,style={"font-size": "14px",'color': 'black'},
                                id=method+"-estimator",persistence=True,persistence_type="memory"),
                            ])
    return component

VarianceThreshold_content=[
    dbc.CardHeader(getAlgoHeader("VarianceThreshold")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            html.Div(dbc.Label("Threshold",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                            html.Div(dbc.Label("Features with a training-set variance lower than this threshold will be removed. The default is to keep all features with non-zero variance, i.e. remove the features that have the same value in all samples.",style={"margin-top": "12px","font-size": "10px",})),
                            dbc.Input(type="number",placeholder="0.0", min=0,id="VarianceThreshold-threshold",persistence=True,persistence_type="memory"),
                    
                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="VarianceThreshold-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("VarianceThreshold"),
                      id="VarianceThreshold-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="VarianceThreshold-collapse",is_open=False,),
    ]

SelectKBest_content=[
    dbc.CardHeader(getAlgoHeader("SelectKBest")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getScoreFun("SelectKBest"),
            dbc.FormGroup([
                            html.Div(dbc.Label("k: number of top features to select",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                            dbc.Input(type="number",placeholder="10", id="SelectKBest-k",min=1,persistence=True,persistence_type="memory"),
                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SelectKBest-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SelectKBest"),
                      id="SelectKBest-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SelectKBest-collapse",is_open=False,),
    ]

SelectPercentile_content=[
    dbc.CardHeader(getAlgoHeader("SelectPercentile")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getScoreFun("SelectPercentile"),
            dbc.FormGroup([
                            html.Div(dbc.Label("percentile: Percent of features to keep.",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                            dbc.Input(type="number",placeholder="10 (%)", id="SelectPercentile-percentile",max=100,min=1,persistence=True,persistence_type="memory"),
                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SelectPercentile-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SelectPercentile"),
                      id="SelectPercentile-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SelectPercentile-collapse",is_open=False,),
    ]

SelectFpr_content=[
    dbc.CardHeader(getAlgoHeader("SelectFpr")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getScoreFun("SelectFpr"),
            dbc.FormGroup([
                            dbc.Label("Alpha",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("The highest p-value for features to be kept.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.05", "value": 0.05},
                                    {"label": "0.1", "value": 0.1},
                                ],value=0.05, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SelectFpr-alpha",persistence=True,persistence_type="memory")
                            ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SelectFpr-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SelectFpr"),
                      id="SelectFpr-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SelectFpr-collapse",is_open=False,),
    ]

SelectFdr_content=[
    dbc.CardHeader(getAlgoHeader("SelectFdr")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getScoreFun("SelectFdr"),
            dbc.FormGroup([
                            dbc.Label("Alpha",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("The highest uncorrected p-value for features to keep.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.05", "value": 0.05},
                                    {"label": "0.1", "value": 0.1},
                                ],value=0.05, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SelectFdr-alpha",persistence=True,persistence_type="memory")
                            ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SelectFdr-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SelectFdr"),
                      id="SelectFdr-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SelectFdr-collapse",is_open=False,),
    ]

SelectFwe_content=[
    dbc.CardHeader(getAlgoHeader("SelectFwe")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getScoreFun("SelectFwe"),
            dbc.FormGroup([
                            dbc.Label("Alpha",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("The highest uncorrected p-value for features to keep.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.05", "value": 0.05},
                                    {"label": "0.1", "value": 0.1},
                                ],value=0.05, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SelectFwe-alpha",persistence=True,persistence_type="memory")
                            ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SelectFwe-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SelectFwe"),
                      id="SelectFwe-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SelectFwe-collapse",is_open=False,),
    ]


RFECV_content=[
    dbc.CardHeader(getAlgoHeader("RFECV")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getEstimator("RFECV"),
            dbc.FormGroup([
                            dbc.Label("Step",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.",style={"margin-top": "12px","font-size": "10px",})),
                            dbc.Input(type="number",placeholder="1",min=1, id="RFECV-step",persistence=True,persistence_type="memory"),
                            
                            
                            dbc.Label("Minimum number of features to be selected",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dbc.Input(type="number",placeholder="1",min=1, id="RFECV-min_features_to_select",persistence=True,persistence_type="memory"),
                        
                        
                        ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="RFECV-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("RFECV"),
                      id="RFECV-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="RFECV-collapse",is_open=False,),
    ]

SelectFromModel_content=[
    dbc.CardHeader(getAlgoHeader("SelectFromModel")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getEstimator("SelectFromModel"),
            dbc.FormGroup([
                            
                            dbc.Label("Maximum number of features to select",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dbc.Input(type="number",placeholder="None",min=1, id="SelectFromModel-max_features",persistence=True,persistence_type="memory"),
                        
                        ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SelectFromModel-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SelectFromModel"),
                      id="SelectFromModel-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SelectFromModel-collapse",is_open=False,),
    ]



SequentialFeatureSelector_content=[
    dbc.CardHeader(getAlgoHeader("SequentialFeatureSelector")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getEstimator("SequentialFeatureSelector"),
            dbc.FormGroup([
                            dbc.Label("Number of features to select",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dbc.Input(type="number",placeholder="Half of the features will be selected by default",min=1, id="SequentialFeatureSelector-n_features_to_select",persistence=True,persistence_type="memory"),
                        
                            dbc.Label("Direction",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("Whether to perform forward selection or backward selection.",style={"margin-top": "12px","font-size": "10px",})),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Forward", "value": "forward"},
                                    {"label": "Backward", "value": "backward"},   
                                ],value="forward",id="SequentialFeatureSelector-direction",
                                persistence=True,persistence_type="memory"),
                        
                        ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SequentialFeatureSelector-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SequentialFeatureSelector"),
                      id="SequentialFeatureSelector-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SequentialFeatureSelector-collapse",is_open=False,),
    ]




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Card Layouts   
featureSelection_content =dbc.Card([
                 dbc.CardBody([                        
                        dbc.Row([
                                dbc.Col(dbc.Card(RFECV_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(SelectFdr_content, color="secondary", outline=True)),
                                ],className="mb-4",),
                        
                        dbc.Row([
                                dbc.Col(dbc.Card(SelectFpr_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(SelectFromModel_content, color="secondary", outline=True)),
                                ],className="mb-4",),   
                        
                        dbc.Row([
                                dbc.Col(dbc.Card(SelectFwe_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(SelectKBest_content, color="secondary", outline=True)),
                                ],className="mb-4",),  
                         
                        dbc.Row([
                                dbc.Col(dbc.Card(SequentialFeatureSelector_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(SelectPercentile_content, color="secondary", outline=True)),
                                ],className="mb-4",), 
                        
                        dbc.Row([
                                dbc.Col(dbc.Card(VarianceThreshold_content, color="secondary", outline=True),width=6),
                                ],className="mb-4",), 


                            ]),
                 
                ],className="mt-3",color="dark", outline=True)


# =============================================================================
# #Card Layouts   
# featureSelection_content =dbc.Card([
#                  dbc.CardBody([
#                         dbc.Row([
#                                 dbc.Col(dbc.Card(VarianceThreshold_content, color="secondary", outline=True)),
#                                 dbc.Col(dbc.Card(SelectKBest_content, color="secondary", outline=True)),
#                                 ],className="mb-4",),
#                         
#                         dbc.Row([
#                                 dbc.Col(dbc.Card(SelectPercentile_content, color="secondary", outline=True)),
#                                 dbc.Col(dbc.Card(SelectFpr_content, color="secondary", outline=True)),
#                                 ],className="mb-4",),   
#                         
#                         dbc.Row([
#                                 dbc.Col(dbc.Card(SelectFdr_content, color="secondary", outline=True)),
#                                 dbc.Col(dbc.Card(SelectFwe_content, color="secondary", outline=True)),
#                                 ],className="mb-4",),  
#                          
#                         dbc.Row([
#                                 dbc.Col(dbc.Card(RFECV_content, color="secondary", outline=True)),
#                                 dbc.Col(dbc.Card(SelectFromModel_content, color="secondary", outline=True)),
#                                 ],className="mb-4",), 
#                         
#                         dbc.Row([
#                                 dbc.Col(dbc.Card(SequentialFeatureSelector_content, color="secondary", outline=True),width=6),
#                                 ],className="mb-4",), 
# 
# 
#                             ]),
#                  
#                 ],className="mt-3",color="dark", outline=True)
# =============================================================================


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#list of all parameters of feature sel algo
from UI.componentIDs import featSel_Com_IDS
featSel_Com_IDS=featSel_Com_IDS.split(",")
featSel_Com_IDS = [sub[1 : ] for sub in featSel_Com_IDS]

#get all the algo names
global algoName
algoName=[]
for item in featSel_Com_IDS:
    if "-" not in item and "_" not in item:
        algoName.append(item)

global paraname
paraname=[]

for item in featSel_Com_IDS:
    #we dont need collapse and info state as of now
    if "collapse" in item or "info" in item:
        continue
    #create a dict with algo name
    if item in algoName:
        continue
    #otherwise save its parameters
    else:
        paraname.append(item)

@app.callback(
    Output("featSel_tab_para", 'data'),
    [Input("{}".format(_), 'value') for _ in algoName],
    [Input("{}".format(_), 'value') for _ in paraname]

    )
def get_featSel_tab_input(*args): 
    
    #args are tupple of list containing boolean
    #([True], [True], [True], [True], [], [True], [], [], [], [], [])
    
    options=args[0:len(algoName)] #all active algo
    para_state=args[len(algoName):len(args)] #all par value

    para_indexer=0 #to track the current position of paraname and pasa state
    data={}         # to save all the data
     
  
    #first iterate through each algo state
    for i in range(0,len(options)):
          
        #check if that algo is active
        if len(options[i])>0:  
  
            data[algoName[i]]={}   #creat a empty list for that algo
             
            #iterate throgh the parastate and paraname simultaneously
            for j in range(para_indexer,len(para_state)):
                #sanity check to avoid wrong paraname and value pair formation
                if algoName[i] not in paraname[j]:
                    break
                
                #assign the para state to correspnding para name
                data[algoName[i]][paraname[j]]= para_state[j]
                para_indexer+=1
                
        else:
            #move the para state list indexer top the next algorithm 
            #parameters using paraname list
            for j in range(para_indexer,len(para_state)):
                if algoName[i] in paraname[j]:
                    para_indexer+=1
                else:
                    break
    return data




genrateInfoCallback("VarianceThreshold")
genrateCollapseCallback("VarianceThreshold")

genrateInfoCallback("SelectKBest")
genrateCollapseCallback("SelectKBest")

genrateInfoCallback("SelectPercentile")
genrateCollapseCallback("SelectPercentile")

genrateInfoCallback("SelectFpr")
genrateCollapseCallback("SelectFpr")

genrateInfoCallback("SelectFdr")
genrateCollapseCallback("SelectFdr")

genrateInfoCallback("SelectFwe")
genrateCollapseCallback("SelectFwe")

genrateInfoCallback("RFECV")
genrateCollapseCallback("RFECV")

genrateInfoCallback("SelectFromModel")
genrateCollapseCallback("SelectFromModel")

genrateInfoCallback("SequentialFeatureSelector")
genrateCollapseCallback("SequentialFeatureSelector")
