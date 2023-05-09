#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:38:23 2021

@author: akshay
"""

import dash_bootstrap_components as dbc
from dash import dcc,html
from UI.scaling import genrateInfoCallback, genrateCollapseCallback,infoText
from app import app
from dash.dependencies import Input, Output,State

#define card for Oversampling tab
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #get the header of each card of sampling tab
def getSamplingHeader(scaling):
    return dbc.Row([
     dbc.Col(dbc.Checklist(options=[{"label": scaling,"value": True}],
                                    value=[],
                                    id=scaling,
                                    inline=True,switch=True,labelStyle={"font-weight": "bold",
                                                                        "font-size": "18px"},
                                    labelCheckedStyle={"color": "green"},persistence=True,persistence_type="memory"),
              width={"size": 9},),
   dbc.Col(dbc.Button("Parameters",id=scaling+"-collapse-button",
            className="mr-1",size="sm",color="light",n_clicks=0,),
        width={"size": 3},)
       ])

   #--------------------------
algorithms_Oversampling = [
    dbc.CardHeader("Algorithms",style={"font-weight": "bold","font-size": "18px"}),
        
    dbc.CardBody(
        [
            dbc.FormGroup([
                        dbc.Checklist(
                            options=[
                                {"label": "ADASYN", "value": "ADASYN"},
                                {"label": "SMOTE", "value": "SMOTE"},
                                {"label": "BorderlineSMOTE", "value": "BorderlineSMOTE"},
                                {"label": "KMeansSMOTE", "value": "KMeansSMOTE"},
                                {"label": "SVMSMOTE", "value": "SVMSMOTE"},
                                {"label": "RandomOverSampler", "value": "RandomOverSampler"},
                            ],id="algorithms_Oversampling",persistence=True,persistence_type="memory"),

                    ],),
            
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="algorithms_Oversampling-info-btn",n_clicks=0,),
            dbc.Alert(html.Label([html.H4("ADASYN"),infoText("ADASYN"),
                                  html.H4("SMOTE"),infoText("SMOTE"),
                                  html.H4("BorderlineSMOTE"),infoText("BorderlineSMOTE"),
                                  html.H4("KMeansSMOTE"),infoText("KMeansSMOTE"),
                                  html.H4("SVMSMOTE"),infoText("SVMSMOTE"),
                                  html.H4("RandomOverSampler"),infoText("RandomOverSampler")],
                                   style={"text-align": "center"}),
                      id="algorithms_Oversampling-info-text",dismissable=True,color="info",is_open=False,),
        ]),

]

   #--------------------------
param_Oversampling = [
    dbc.CardHeader("Parameters",style={"font-weight": "bold","font-size": "18px"}),
        
    dbc.CardBody(
        [
            dbc.FormGroup([
                        dbc.Label("Sampling Strategy",style={"font-weight": "bold","font-size": "18px"}),
                        dcc.Dropdown(
                            options=[
                                {"label": "Minority: resample only the minority class", "value": "minority"},
                                {"label": "Not minority: resample all classes but the minority class", "value": "not minority"},
                                {"label": "Not majority: resample all classes but the majority class", "value": "not majority"},
                                {"label": "All: resample all classes", "value": "all"},
                                {"label": "Auto: equivalent to Not majority", "value": "auto"},

                            ],value='auto', clearable=False,style={'color': 'black'},
                            id="samp_strat_param_Oversampling",persistence=True,persistence_type="memory"),
                        
                        html.Div(dbc.Label("K Neighbors",style={"margin-top": "15px","font-weight": "bold",})),
                        html.Div(dbc.Label("Number of nearest neighbours to used to construct synthetic samples. ",style={"margin-top": "12px","font-size": "12px",})),

                        dbc.Input(type="number",placeholder="5", min=1,id="k_neighbors_param_Oversampling",persistence=True,persistence_type="memory"),
                    

                    ],),
            
            
# =============================================================================
#             dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="param_Oversampling-info-btn",n_clicks=0,),
#             dbc.Alert("Hello! I am an alert",
#                       id="param_Oversampling-info-text",dismissable=True,color="info",is_open=False,),
# =============================================================================
        ]),    

]


    #content for Oversampling sub-tab
    #--------------------------
oversampling_content = dbc.CardBody([
                        dbc.Row([
                                dbc.Col(dbc.Card(algorithms_Oversampling, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(param_Oversampling, color="secondary", outline=True)),
                                
                                ],className="mb-4",),
            
                            ])



    #get Oversampling sub-tab data
    #--------------------------
@app.callback(
    Output("overSamp_tab_para", 'data'),
    [Input("algorithms_Oversampling", 'value')],
    [Input("samp_strat_param_Oversampling", 'value')],
    [Input("k_neighbors_param_Oversampling", 'value')]
    )
def get_scale_tab_algo_input(*args):
    algorithms=args[0] #all active algo
    sampling_strategy=args[1]
    k_neighbors=args[2]
    
    data={}
    
    if algorithms!=None:
        for algo in algorithms:
            data[algo]={}
            data[algo]["sampling_strategy"]=sampling_strategy
            
            if k_neighbors!=None:
                data[algo]["k_neighbors"]=k_neighbors
    
    return data

    
#define card for Undersampling tab
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #--------------------------
    #common parameters for all algo
def getSampStratPara(id):
    component= dbc.FormGroup([
                        dbc.Label("Sampling Strategy",style={"font-weight": "bold","font-size": "18px"}),
                        dcc.Dropdown(
                            options=[
                                {"label": "Minority: resample only the minority class", "value": "minority"},
                                {"label": "Not minority: resample all classes but the minority class", "value": "not minority"},
                                {"label": "Not majority: resample all classes but the majority class", "value": "not majority"},
                                {"label": "All: resample all classes", "value": "all"},
                                {"label": "Auto: equivalent to Not majority", "value": "auto"},

                            ],value='auto', clearable=False,style={'color': 'black'},
                            id=id+"-sampling_strategy",persistence=True,persistence_type="memory")
                        ])
    return component



def get_neighbors_Para(label,intro,id,default):
    component= dbc.FormGroup([
                        html.Div(dbc.Label(label,style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                        html.Div(dbc.Label(intro,style={"font-size": "12px",})),
                        dbc.Input(type="number",min=2, placeholder=default,id=id,persistence=True,persistence_type="memory"),
                        ])
    return component



def get_kind_sel_Para(id):
    component= dbc.FormGroup([
        
                        dbc.Label("Strategy to use in order to exclude samples",style={"font-weight": "bold","font-size": "18px"}),
                        dcc.Dropdown(
                            options=[
                                {"label": "All: all neighbours will have to agree with the samples of interest to not be excluded.", "value": "all"},
                                {"label": "Mode: the majority vote of the neighbours will be used in order to exclude a sample.", "value": "mode"},
                            ],value='all', clearable=False,style={"font-size": "14px",'color': 'black'},
                            id=id+"-kind_sel",persistence=True,persistence_type="memory")
                        ])

    return component
    
    #--------------------------
AllKNN_content = [
    dbc.CardHeader(getSamplingHeader("AllKNN")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("AllKNN"), 
                            get_neighbors_Para("N Neighbors",
                                               "Size of the neighbourhood to consider to compute the nearest neighbors.",
                                               "AllKNN-n_neighbors","3"),
                            get_kind_sel_Para("AllKNN"),
                            html.Div(dbc.Label("Allow Minority",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                            html.Div(dbc.Label("Allows the majority classes to become the minority class without early stopping.",style={"font-size": "12px",})),
                            dbc.RadioItems(options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False}
                                    ],value=False,id="AllKNN-allow_minority"
                                    ,persistence=True,persistence_type="memory")
                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="AllKNN-info-btn",n_clicks=0,),
            dbc.Alert(infoText("AllKNN"),
                      id="AllKNN-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="AllKNN-collapse",is_open=False,),
    ]
 
    
    #--------------------------
ClusterCentroids_content = [
    dbc.CardHeader(getSamplingHeader("ClusterCentroids")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("ClusterCentroids"), 

                            html.Div(dbc.Label("Voting strategy to generate the new samples",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "Hard: the nearest-neighbors of the centroids found using the clustering algorithm will be used.", "value": "hard"},
                                    {"label": "Soft: the centroids found by the clustering algorithm will be used.", "value": "soft"},
                                    {"label": "Auto: if the input is sparse, it will default on 'hard' otherwise, 'soft' will be used.", "value": "auto"},

                                ],value='auto', clearable=False,style={"font-size": "14px"},
                                id="ClusterCentroids-voting",persistence=True,persistence_type="memory")
                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="ClusterCentroids-info-btn",n_clicks=0,),
            dbc.Alert(infoText("ClusterCentroids"),
                      id="ClusterCentroids-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="ClusterCentroids-collapse",is_open=False,),
    ]

    #--------------------------
CondensedNearestNeighbour_content = [
    dbc.CardHeader(getSamplingHeader("CondensedNearestNeighbour")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("CondensedNearestNeighbour"), 
                            get_neighbors_Para("N Neighbors",
                                               "Size of the neighbourhood to consider to compute the nearest neighbors.",
                                               "CondensedNearestNeighbour-n_neighbors","None"),
                            get_neighbors_Para("N seeds S",
                                               "Number of samples to extract in order to build the set S.",
                                               "CondensedNearestNeighbour-n_seeds_S","default=1"),

                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="CondensedNearestNeighbour-info-btn",n_clicks=0,),
            dbc.Alert(infoText("CondensedNearestNeighbour"),
                      id="CondensedNearestNeighbour-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="CondensedNearestNeighbour-collapse",is_open=False,),
    ]

    #--------------------------
EditedNearestNeighbours_content = [
    dbc.CardHeader(getSamplingHeader("EditedNearestNeighbours")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("EditedNearestNeighbours"), 
                            get_neighbors_Para("N Neighbors",
                                               "Size of the neighbourhood to consider to compute the nearest neighbors.",
                                               "EditedNearestNeighbours-n_neighbors","3"),
                            get_kind_sel_Para("EditedNearestNeighbours"),
                        ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="EditedNearestNeighbours-info-btn",n_clicks=0,),
            dbc.Alert(infoText("EditedNearestNeighbours"),
                      id="EditedNearestNeighbours-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="EditedNearestNeighbours-collapse",is_open=False,),
    ]

    #--------------------------
RepeatedEditedNearestNeighbours_content = [
    dbc.CardHeader(getSamplingHeader("RepeatedEditedNearestNeighbours")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("RepeatedEditedNearestNeighbours"), 
                            get_neighbors_Para("N Neighbors",
                                               "Size of the neighbourhood to consider to compute the nearest neighbors.",
                                               "RepeatedEditedNearestNeighbours-n_neighbors",3),
                            get_kind_sel_Para("RepeatedEditedNearestNeighbours"),
                        ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="RepeatedEditedNearestNeighbours-info-btn",n_clicks=0,),
            dbc.Alert(infoText("RepeatedEditedNearestNeighbours"),
                      id="RepeatedEditedNearestNeighbours-info-text",dismissable=True,color="info",is_open=False,),
        ]),
        id="RepeatedEditedNearestNeighbours-collapse",is_open=False,),

    ]

    #--------------------------
InstanceHardnessThreshold_content = [
    dbc.CardHeader(getSamplingHeader("InstanceHardnessThreshold")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("InstanceHardnessThreshold"),
                            html.Div(dbc.Label("CV",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                            html.Div(dbc.Label("Number of folds to be used when estimating samples’ instance hardness.",style={"font-size": "12px",})),
                            dbc.Input(type="number", min=3,placeholder=5,id="InstanceHardnessThreshold-cv",persistence=True,persistence_type="memory"),

                        ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="InstanceHardnessThreshold-info-btn",n_clicks=0,),
            dbc.Alert(infoText("InstanceHardnessThreshold"),
                      id="InstanceHardnessThreshold-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="InstanceHardnessThreshold-collapse",is_open=False,),
    ]
 


    #--------------------------
NearMiss_content = [
    dbc.CardHeader(getSamplingHeader("NearMiss")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("NearMiss"), 
                            get_neighbors_Para("N Neighbors",
                                               "Size of the neighbourhood to consider to compute the average distance to the minority point samples.",
                                               "NearMiss-n_neighbors","3"),

                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="NearMiss-info-btn",n_clicks=0,),
            dbc.Alert(infoText("NearMiss"),
                      id="NearMiss-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="NearMiss-collapse",is_open=False,),
    ]

    #--------------------------
NeighbourhoodCleaningRule_content = [
    dbc.CardHeader(getSamplingHeader("NeighbourhoodCleaningRule")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("NeighbourhoodCleaningRule"), 
                            get_neighbors_Para("N Neighbors",
                                               "Size of the neighbourhood to consider to compute the nearest neighbors.",
                                               "NeighbourhoodCleaningRule-n_neighbors","3"),
                            get_kind_sel_Para("NeighbourhoodCleaningRule"),
                            html.Div(dbc.Label("Threshold Cleaning",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                            html.Div(dbc.Label("Threshold used to whether consider a class or not during the cleaning after applying ENN.",style={"font-size": "12px",})),
                            dbc.Input(type="number", placeholder="0.5",min=0.1,id="NeighbourhoodCleaningRule-threshold_cleaning",persistence=True,persistence_type="memory"),


                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="NeighbourhoodCleaningRule-info-btn",n_clicks=0,),
            dbc.Alert(infoText("NeighbourhoodCleaningRule"),
                      id="NeighbourhoodCleaningRule-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="NeighbourhoodCleaningRule-collapse",is_open=False,),
    ]


    #--------------------------
OneSidedSelection_content = [
    dbc.CardHeader(getSamplingHeader("OneSidedSelection")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("OneSidedSelection"), 
                            get_neighbors_Para("N Neighbors",
                                               "Size of the neighbourhood to consider to compute the nearest neighbors.",
                                               "OneSidedSelection-n_neighbors","None"),
                             get_neighbors_Para("N seeds S",
                                               "Number of samples to extract in order to build the set S.",
                                               "OneSidedSelection-n_seeds_S","1"),


                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="OneSidedSelection-info-btn",n_clicks=0,),
            dbc.Alert(infoText("OneSidedSelection"),
                      id="OneSidedSelection-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="OneSidedSelection-collapse",is_open=False,),
    ]



    #--------------------------
RandomUnderSampler_content = [
    dbc.CardHeader(getSamplingHeader("RandomUnderSampler")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("RandomUnderSampler"), 
                            
                            dbc.Label("Replacement",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("Whether the sample is with or without replacement.",style={"font-size": "12px",})),
                            dbc.RadioItems(options=[
                                            {"label": "True", "value": True},
                                            {"label": "False", "value": False}
                                ],value=False,id="RandomUnderSampler-replacement"
                                ,persistence=False,persistence_type="memory"),

                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="RandomUnderSampler-info-btn",n_clicks=0,),
            dbc.Alert(infoText("RandomUnderSampler"),
                      id="RandomUnderSampler-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="RandomUnderSampler-collapse",is_open=False,),
    ]

    #--------------------------
TomekLinks_content = [
    dbc.CardHeader(getSamplingHeader("TomekLinks")),
        
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            getSampStratPara("TomekLinks"), 

                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="TomekLinks-info-btn",n_clicks=0,),
            dbc.Alert(infoText("TomekLinks"),
                      id="TomekLinks-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="TomekLinks-collapse",is_open=False,),
    ]




    #content for Undersampling sub-tab
    #--------------------------
undersampling_content =dbc.CardBody([
                        dbc.Row([
                                dbc.Col(dbc.Card(AllKNN_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(ClusterCentroids_content, color="secondary", outline=True)),

                                ],className="mb-4",),                        
                        dbc.Row([
                                dbc.Col(dbc.Card(CondensedNearestNeighbour_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(EditedNearestNeighbours_content, color="secondary", outline=True)),

                                ],className="mb-4",),
                        dbc.Row([
                                dbc.Col(dbc.Card(InstanceHardnessThreshold_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(NearMiss_content, color="secondary", outline=True)),

                                ],className="mb-4",),            
                        dbc.Row([
                                dbc.Col(dbc.Card(NeighbourhoodCleaningRule_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(OneSidedSelection_content, color="secondary", outline=True)),

                                ],className="mb-4",), 
                        dbc.Row([
                                dbc.Col(dbc.Card(RandomUnderSampler_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(RepeatedEditedNearestNeighbours_content, color="secondary", outline=True)),

                                ],className="mb-4",),
                        
                        dbc.Row([
                                dbc.Col(dbc.Card(TomekLinks_content, color="secondary", outline=True),width=6),
                                ],className="mb-4",),
                            ])




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#list of all parameters of undersampling algo
from UI.componentIDs import undersampling_Com_IDS
undersampling_Com_IDS=undersampling_Com_IDS.split(",")
undersampling_Com_IDS = [sub[1 : ] for sub in undersampling_Com_IDS]

#get all the algo names
global algoName
algoName=[]
for item in undersampling_Com_IDS:
    if "-" not in item and "_" not in item:
        algoName.append(item)

global paraname
paraname=[]

for item in undersampling_Com_IDS:
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
    Output("underSamp_tab_para", 'data'),
    [Input("{}".format(_), 'value') for _ in algoName],
    [Input("{}".format(_), 'value') for _ in paraname]
    )
def get_underSamp_tab_input(*args): 
    
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
                
    
    #create a list of active algos
# =============================================================================
#     algos=",".join((str(val[0]) for val in options if val))
#     algos_list =[s for s in algos.split(',')]
# =============================================================================
# =============================================================================
#     algos_list = list(filter(None, algos_list))
#     
#     print(algos_list)
#     if len(algos_list)>0:
#         
#         #dict to store active algos and respective features
#         active_scalingAlgo={k:[] for k in algos_list}
#         
#         #assign all para to its key using scalingAlgo
#         i=0
#         for key in scalingAlgo.keys():
#             for par in scalingAlgo[key]:
#                 scalingAlgo[key][par]=para[i]
#                 i+=1
#         
#         #fetch para of active algo only
#         for key in active_scalingAlgo.keys():
#             active_scalingAlgo[key]=scalingAlgo[key]
#         print(active_scalingAlgo)
#        
#         data=active_scalingAlgo
# =============================================================================
    


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#calllbacks of dataSampling info and collapse


genrateInfoCallback("algorithms_Oversampling")
#genrateInfoCallback("param_Oversampling")

genrateInfoCallback("AllKNN")
genrateCollapseCallback("AllKNN")

genrateInfoCallback("ClusterCentroids")
genrateCollapseCallback("ClusterCentroids")

genrateInfoCallback("CondensedNearestNeighbour")
genrateCollapseCallback("CondensedNearestNeighbour")

genrateInfoCallback("EditedNearestNeighbours")
genrateCollapseCallback("EditedNearestNeighbours")

genrateInfoCallback("RepeatedEditedNearestNeighbours")
genrateCollapseCallback("RepeatedEditedNearestNeighbours")

genrateInfoCallback("InstanceHardnessThreshold")
genrateCollapseCallback("InstanceHardnessThreshold")

genrateInfoCallback("NearMiss")
genrateCollapseCallback("NearMiss")

genrateInfoCallback("NeighbourhoodCleaningRule")
genrateCollapseCallback("NeighbourhoodCleaningRule")

genrateInfoCallback("OneSidedSelection")
genrateCollapseCallback("OneSidedSelection")

genrateInfoCallback("RandomUnderSampler")
genrateCollapseCallback("RandomUnderSampler")

genrateInfoCallback("TomekLinks")
genrateCollapseCallback("TomekLinks")


