#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:40:48 2022

@author: akshay
"""

from dash import dcc,html
import dash_bootstrap_components as dbc
from app import app
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
from dash import dcc,html
from UI.scaling import genrateInfoCallback, genrateCollapseCallback,genrateAlertCallback,infoText
from UI.dataSampling import get_neighbors_Para
from app import app
from dash.dependencies import Input, Output,State

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Metrices

modelEval_metrics=dbc.FormGroup([
                            
                dbc.Label("Metrics",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                dcc.Dropdown(
                            options=[
                                {"label": "Accuracy", "value": "accuracy"},
                                {"label": "Average Precision", "value": "average_precision"},          
                                {"label": "Balanced Accuracy", "value": "balanced_accuracy"},
                                {"label": "Brier score loss", "value": "neg_brier_score"},
                                {"label": "F1", "value": "f1"},
                                {"label": "F1 Micro", "value": "f1_micro"},
                                {"label": "F1 Macro", "value": "f1_macro"},                                
                                {"label": "F1 Weighted", "value": "f1_weighted"},
                                {"label": "Jaccard", "value": "jaccard"},
                                {"label": "Matthews correlation coefficient (MCC)", "value": "matthews_corrcoef"},
                                {"label": "Precision", "value": "precision"},
                                {"label": "Recall", "value": "recall"},
                                {"label": "Roc Auc", "value": "roc_auc"},
                                {"label": "Top k Accuracy", "value": "top_k_accuracy"},

                                
                            ],value=[],
                            multi=True,clearable=False,style={"font-size": "14px",'color': 'green'},
                            id="modelEval_metrics",persistence=True,persistence_type="memory"),
                
                dbc.Label("Refit Metric",style={"margin-top": "20px","font-weight": "bold","font-size": "18px"}),
                html.Div(dbc.Label("For multiple metric evaluation, this needs to be a str denoting the scorer that would be used to find the best parameters for refitting the estimator at the end.",style={"margin-top": "10px","font-size": "10px",})),
                dcc.Dropdown(
                            options=[
                                {"label": "Accuracy", "value": "accuracy"},
                                {"label": "Balanced Accuracy", "value": "balanced_accuracy"},
                                {"label": "Top k Accuracy", "value": "top_k_accuracy"},
                                {"label": "Average Precision", "value": "average_precision"},          
                                {"label": "Brier score loss", "value": "neg_brier_score"},
                                {"label": "F1", "value": "f1"},
                                {"label": "F1 Micro", "value": "f1_micro"},
                                {"label": "F1 Macro", "value": "f1_macro"},                                
                                {"label": "F1 Weighted", "value": "f1_weighted"},
                                {"label": "Precision", "value": "precision"},
                                {"label": "Recall", "value": "recall"},
                                {"label": "Jaccard", "value": "jaccard"},
                                {"label": "Roc Auc", "value": "roc_auc"},
                                
                            ],value=[],
                            multi=False,clearable=False,style={"font-size": "14px",'color': 'green'},
                            id="refit_metric",persistence=True,persistence_type="memory"),
                
                html.Br(),       
                dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="metric-info-btn",n_clicks=0,color="warning"),
                html.Br(),
                dbc.Alert(
                          html.Label(["Except for Brier score loss and Matthews correlation coefficient (MCC), all evaluation metrics range from 0 to 1, with 0 being the worst and 1 being the best.",html.Br(),
                                      html.Br(),"On the other hand, Brier score loss and MCC range between -1 and +1. A value of +1 represents a perfect prediction, 0 is an average random prediction, and -1 is an inverse prediction. To simplify comparisons between metrics and for better visualization, we have scaled the Brier score loss and MCC between 0 and 1 from -1 to +1. Therefore, +1 represents a perfect prediction, 0.5 represents an average random prediction, and 0 represents an inverse prediction.",
                                   ],style={"text-align": "Justify"}),
                      id="metric-info-text",dismissable=True,is_open=False,color="info")
                               
                      ])               
  
#functions to be used for cards
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!              
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

#get the Integer input of model eval tab
def getSplitDiv(id):
    component= dbc.FormGroup([
                 
                    html.Div(dbc.Label("Number of folds",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    html.Div(dbc.Label("Must be at least 2.",style={"margin-top": "12px","font-size": "10px",})),
                    dbc.Input(type="number",placeholder=5, min=2,id=id,persistence=True,persistence_type="memory"),
                    ])

    return component

#get the Integer input of model eval tab
def getRepeatDiv(id):
    component= dbc.FormGroup([
                
                    html.Div(dbc.Label("Number of Repeats",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    html.Div(dbc.Label("Number of times cross-validator needs to be repeated.",style={"margin-top": "12px","font-size": "10px",})),
                    dbc.Input(type="number",placeholder=10, min=1,id=id,persistence=True,persistence_type="memory"),
                    ])

    return component


        #get the boolean input of model eval tab
def getBoolDiv(id):
    component= dbc.FormGroup([
                
                    html.Div(dbc.Label("Shuffle",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    html.Div(dbc.Label("Whether to shuffle the data before splitting into batches.",style={"margin-top": "12px","font-size": "10px",})),
                    dbc.RadioItems(options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False}
                        ],value=False,id=id
                        ,persistence=True,persistence_type="memory")
                    ])

    return component

            

#define card for Methods tab
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

KFold_content=[
    dbc.CardHeader(getAlgoHeader("KFold")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getSplitDiv("KFold-n_splits"),
            getBoolDiv("KFold-shuffle"),

            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="KFold-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("KFold"),
                      id="KFold-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="KFold-collapse",is_open=False,),
    ]

StratifiedKFold_content=[
    dbc.CardHeader(getAlgoHeader("StratifiedKFold")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getSplitDiv("StratifiedKFold-n_splits"),
            getBoolDiv("StratifiedKFold-shuffle"),

            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="StratifiedKFold-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("StratifiedKFold"),
                      id="StratifiedKFold-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="StratifiedKFold-collapse",is_open=False,),
    ]

RepeatedKFold_content=[
    dbc.CardHeader(getAlgoHeader("RepeatedKFold")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getSplitDiv("RepeatedKFold-n_splits"),
            getRepeatDiv("RepeatedKFold-n_repeats"),

            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="RepeatedKFold-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("RepeatedKFold"),
                      id="RepeatedKFold-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="RepeatedKFold-collapse",is_open=False,),
    ]


RepeatedStratifiedKFold_content=[
    dbc.CardHeader(getAlgoHeader("RepeatedStratifiedKFold")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            getSplitDiv("RepeatedStratifiedKFold-n_splits"),
            getRepeatDiv("RepeatedStratifiedKFold-n_repeats"),

            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="RepeatedStratifiedKFold-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("RepeatedStratifiedKFold"),
                      id="RepeatedStratifiedKFold-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="RepeatedStratifiedKFold-collapse",is_open=False,),
    ]

LeaveOneOut_content=[
    dbc.CardHeader(getAlgoHeader("LeaveOneOut")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="LeaveOneOut-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("LeaveOneOut"),
                      id="LeaveOneOut-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="LeaveOneOut-collapse",is_open=False,),
    ]

LeavePOut_content=[
    dbc.CardHeader(getAlgoHeader("LeavePOut")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            html.Div(dbc.Label("p",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("Size of the test sets. Must be strictly less than the number of samples.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.Input(type="number",value=10, min=10,id="LeavePOut-p",persistence=True,persistence_type="memory"),


            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="LeavePOut-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("LeavePOut"),
                      id="LeavePOut-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="LeavePOut-collapse",is_open=False,),
    ]

ShuffleSplit_content=[
    dbc.CardHeader(getAlgoHeader("ShuffleSplit")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            
            html.Div(dbc.Label("Test Size",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("Represent the proportion of the dataset to include in the test split. If None, the value is set to the complement of the train size. If train size is also None, it will be set to 0.1.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.Input(type="number",placeholder="None", min=0.1,max=1.0,id="ShuffleSplit-test_size",persistence=True,persistence_type="memory"),

            html.Div(dbc.Label("Train Size",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("Represent the proportion of the dataset to include in the train split. If None, the value is set to the complement of the test size. If test size is also None, it will be set to 0.1.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.Input(type="number",placeholder="None", min=0.1,max=1.0,id="ShuffleSplit-train_size",persistence=True,persistence_type="memory"),   
                    
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="ShuffleSplit-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("ShuffleSplit"),
                      id="ShuffleSplit-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="ShuffleSplit-collapse",is_open=False,),
    ]

StratifiedShuffleSplit_content=[
    dbc.CardHeader(getAlgoHeader("StratifiedShuffleSplit")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            
            html.Div(dbc.Label("Test Size",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("Represent the proportion of the dataset to include in the test split. If None, the value is set to the complement of the train size. If train size is also None, it will be set to 0.1.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.Input(type="number",placeholder="None", min=0.1,max=1.0,id="StratifiedShuffleSplit-test_size",persistence=True,persistence_type="memory"),

            html.Div(dbc.Label("Train Size",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("Represent the proportion of the dataset to include in the train split. If None, the value is set to the complement of the test size. If test size is also None, it will be set to 0.1.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.Input(type="number",placeholder="None", min=0.1,max=1.0,id="StratifiedShuffleSplit-train_size",persistence=True,persistence_type="memory"),   
                    
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="StratifiedShuffleSplit-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("StratifiedShuffleSplit"),
                      id="StratifiedShuffleSplit-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="StratifiedShuffleSplit-collapse",is_open=False,),
    ]

NestedCV_content=[
    dbc.CardHeader(getAlgoHeader("NestedCV")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            
            html.Div(dbc.Label("NOTE: StratifiedKFold method will be used as inner and outer loop of the nested-cross validation procedure.",style={"margin-top": "12px","font-size": "10px",})),

            getSplitDiv("NestedCV-n_splits"),
            getBoolDiv("NestedCV-shuffle"),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="NestedCV-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("NestedCV"),
                      id="NestedCV-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="NestedCV-collapse",is_open=False,),
    ]


indeptestset_content=[
    dbc.CardHeader(getAlgoHeader("Independent Test Set")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            
            html.Div(dbc.Label("Test Dataset Size",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("Represent the proportion of the dataset to include in the test split. Value should be bewteen 0 to 0.5.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.Input(type="number",value=0.25, min=0.1,max=0.5,id="Independent Test Set-test_size",persistence=True,persistence_type="memory"),
   
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="Independent Test Set-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(html.Label("The Test dataset provides the gold standard used to evaluate the model. It is only used once a model is completely trained and evaluated with the training and validation datasets using one of the above selected methods.",
                                   style={"text-align": "Justify"}),
                      id="Independent Test Set-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="Independent Test Set-collapse",is_open=False,),
    ]


   
    
   
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Card Layouts   
modelEval_methods =dbc.Card([
                 dbc.CardBody([
                        dbc.Row([
                                dbc.Col(dbc.Card(KFold_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(StratifiedKFold_content, color="secondary", outline=True)),
                                ],className="mb-4",),
                        

                        dbc.Row([

                                  dbc.Col(dbc.Card(RepeatedKFold_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(RepeatedStratifiedKFold_content, color="secondary", outline=True),),
                                ],className="mb-4",),
                        
                        
                        dbc.Row([

                                  dbc.Col(dbc.Card(LeaveOneOut_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(LeavePOut_content, color="secondary", outline=True),),
                                ],className="mb-4",),
  
                        dbc.Row([

                                  dbc.Col(dbc.Card(ShuffleSplit_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(StratifiedShuffleSplit_content, color="secondary", outline=True),),
                                ],className="mb-4",),
        
                        dbc.Row([

                                  dbc.Col(dbc.Card(NestedCV_content, color="secondary", outline=True),width=6),
                                  dbc.Col(dbc.Card(indeptestset_content, color="secondary", outline=True)),
                                ],className="mb-4",),
                        
                            ]),
                 
                 

                ],className="mt-3",color="dark", outline=True)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#list of all parameters of model eval method tabs
from UI.componentIDs import modelEval_Com_IDS
modelEval_Com_IDS=modelEval_Com_IDS.split(",")
modelEval_Com_IDS = [sub[1 : ] for sub in modelEval_Com_IDS]

#get all the algo names
global algoName
algoName=[]
for item in modelEval_Com_IDS:
    if "-" not in item and "_" not in item:
        algoName.append(item)

global paraname
paraname=[]

for item in modelEval_Com_IDS:
    #we dont need collapse and info state as of now
    if "collapse" in item or "info" in item or "alert" in item:
        continue
    #create a dict with algo name
    if item in algoName:
        continue
    #otherwise save its parameters
    else:
        paraname.append(item)
        
        
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@app.callback(
    Output("modelEval_tab_para", 'data'),
    [Input("{}".format(_), 'value') for _ in algoName],
    [Input("{}".format(_), 'value') for _ in paraname],

    )
def get_classification_tab_input(*args): 
    
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

            data[algoName[i]]={}   #create a empty list for that algo
             
            #iterate throgh the parastate and paraname simultaneously
            for j in range(para_indexer,len(para_state)):
                #sanity check to avoid wrong paraname and value pair formation
                algoName_temp=paraname[j].split("-")[0] #extract first part of paraname that is algo name
                if algoName[i] != algoName_temp:
                    break
                
                #assign the para state to correspnding para name
                data[algoName[i]][paraname[j]]= para_state[j]
                para_indexer+=1
                
        else:
            #move the para state list indexer top the next algorithm 
            #parameters using paraname list
            for j in range(para_indexer,len(para_state)):
                algoName_temp=paraname[j].split("-")[0]

                if algoName[i] == algoName_temp:
                    para_indexer+=1
                else:  
                    break
    return data

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@app.callback(
    Output("indepTestSet", 'data'),
    [Input("Independent Test Set", 'value') ,
    Input("Independent Test Set-test_size", 'value')],

    )
def get_IndepTest(active,size): 
    #check if that algo is active
    if len(active)>0 and active[0]==True: 
        return size

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@app.callback(
    Output("modelEval_metrices_tab_para", 'data'),
    [Input(component_id='modelEval_metrics', component_property='value')],

    )
def get_classification_tab_input(*args): 
    return args   

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@app.callback(
    Output("refit_Metric", 'data'),
    [Input(component_id='refit_metric', component_property='value')],

    )
def get_classification_tab_input(*args): 
    return args   

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#calllbacks of scaling info
genrateInfoCallback("KFold")
genrateCollapseCallback("KFold")

genrateInfoCallback("StratifiedKFold")
genrateCollapseCallback("StratifiedKFold")


genrateInfoCallback("RepeatedKFold")
genrateCollapseCallback("RepeatedKFold")


genrateInfoCallback("RepeatedStratifiedKFold")
genrateCollapseCallback("RepeatedStratifiedKFold")

genrateInfoCallback("LeaveOneOut")
genrateCollapseCallback("LeaveOneOut")

genrateInfoCallback("LeavePOut")
genrateCollapseCallback("LeavePOut")

genrateInfoCallback("ShuffleSplit")
genrateCollapseCallback("ShuffleSplit")

genrateInfoCallback("StratifiedShuffleSplit")
genrateCollapseCallback("StratifiedShuffleSplit")

genrateInfoCallback("NestedCV")
genrateCollapseCallback("NestedCV")

genrateInfoCallback("Independent Test Set")
genrateCollapseCallback("Independent Test Set")

genrateInfoCallback("metric")
genrateCollapseCallback("metric")

