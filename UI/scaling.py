#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 17:17:42 2021

@author: akshay
"""
import dash_bootstrap_components as dbc
from dash import dcc,html
from app import app
from dash.dependencies import Input, Output,State

#global scalingAlgo
scalingAlgo={k:[] for k in ["MaxAbs Scaler","MinMax Scaler","Normalizer","PowerTransformer",
             "QuantileTransformer","Robust Scaler","Standard Scaler"]}


#get info text
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from UI.componentIDs import allIds

def infoText(id_):
    text=allIds[id_][0]
    link=allIds[id_][1]
    return html.Label([text,html.Br() ,html.Br() ," Please refer to the ",
                               html.A('scikit-learn user guide', href=link,target="_blank",style={"color": "black"}),
                               " for further details."],
                                   style={"text-align": "Justify"})
#define card for scaling tab
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    #get the header of each card of scaling tab
def getScalingHeader(scaling):
    return dbc.Row([
     dbc.Col(dbc.Checklist(options=[{"label": scaling,"value": scaling}],
                                    value=[],
                                    id=scaling+"-checklist-algo",
                                    inline=True,switch=True,labelStyle={"font-weight": "bold",
                                                                        "font-size": "18px"},
                                    labelCheckedStyle={"color": "green"},persistence=True,persistence_type="memory"),
              width={"size": 8},),
   dbc.Col(dbc.Button("Parameters",id=scaling+"-collapse-button",
            className="mr-1",size="sm",color="light",n_clicks=0,),
        width={"size": 4},)
       ])

    #--------------------------
scalingAlgo["MaxAbs Scaler"]={}

maxAbs_Scaler_content =html.Div( [
    dbc.CardHeader(getScalingHeader("MaxAbs Scaler")),
        
    dbc.Collapse(
     dbc.CardBody(
        [           
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="MaxAbs Scaler-info-btn",n_clicks=0,),
            dbc.Alert(infoText("MaxAbs Scaler"),
                      id="MaxAbs Scaler-info-text",dismissable=True,color="info",is_open=False,),
        ]),
         id="MaxAbs Scaler-collapse",is_open=False,),

])



    #--------------------------
scalingAlgo["MinMax Scaler"]={"feature_range":""}

minmax_Scaler_content = [
    dbc.CardHeader(getScalingHeader("MinMax Scaler")),
    
    dbc.Collapse(
        dbc.CardBody(
        [
            dbc.FormGroup([
                            html.Div(dbc.Label("Feature Range",style={"font-weight": "bold","font-size": "18px"})),
                            dcc.RangeSlider( id="MinMax Scaler-feature_range", min=-2, max=2, value=[0, 1],
                                            tooltip={"always_visible": True,"placement":"top"},
                                            marks={0:"Desired range of transformed data."},persistence=True,persistence_type="memory")
                            ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="MinMax Scaler-info-btn",n_clicks=0,),
            dbc.Alert(infoText("MinMax Scaler"),
                      id="MinMax Scaler-info-text",dismissable=True,color="info",is_open=False,),
        ]),
        id="MinMax Scaler-collapse",is_open=False,),
]

    #--------------------------
scalingAlgo["Normalizer"]={"norm":""}

normalizer_Scaler_content = [
    dbc.CardHeader(getScalingHeader("Normalizer")),
        
    dbc.Collapse(
    dbc.CardBody(
        [ 
            dbc.FormGroup([

                    html.Div(dbc.Label("Norm Type",style={"font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "l1", "value": "l1"},
                                    {"label": "l2", "value": "l2"},
                                    {"label": "max", "value": "max"},
                        ],value="l2",id="Normalizer-norm",persistence=True,persistence_type="memory"),
                ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="Normalizer-info-btn",n_clicks=0,),
            dbc.Alert(infoText("Normalizer"),
                      id="Normalizer-info-text",dismissable=True,color="info",is_open=False,),
        ]),
    id="Normalizer-collapse",is_open=False,),
]


    #--------------------------
scalingAlgo["PowerTransformer"]={"method":"","standardize":""}

powerTransformer_Scaler_content = [
    dbc.CardHeader(getScalingHeader("PowerTransformer")),
        
    dbc.Collapse(
    dbc.CardBody(
        [ 
            dbc.FormGroup([

                    html.Div(dbc.Label("Power Transform Method",style={"font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "yeo-johnson", "value": "yeo-johnson"},
                                    {"label": "box-cox", "value": "box-cox"}
                        ],value="yeo-johnson",id="PowerTransformer-method",
                        persistence=True,persistence_type="memory",),
                    
                    html.Div(dbc.Label("Standardize",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False}
                        ],value=True,id="PowerTransformer-standardize"
                        ,persistence=True,persistence_type="memory")
                    
                ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="PowerTransformer-info-btn",n_clicks=0,),
            dbc.Alert(infoText("PowerTransformer"),
                      id="PowerTransformer-info-text",dismissable=True,color="info",is_open=False,),
        ]),
    id="PowerTransformer-collapse",is_open=False,),
]


    #--------------------------
scalingAlgo["QuantileTransformer"]={"n_quantiles":"","output_distribution":""}

quantileTransformer_Scaler_content = [
    dbc.CardHeader(getScalingHeader("QuantileTransformer")),
        
    dbc.Collapse(
    dbc.CardBody(
        [ 
            dbc.FormGroup([

                    html.Div(dbc.Label("Number of quantiles",style={"font-weight": "bold","font-size": "18px"})),
                    dbc.Input(type="number",min=0,placeholder="default=1000", id="QuantileTransformer-n_quantiles",persistence=True,persistence_type="memory"),
                    
                    html.Div(dbc.Label("Output Distribution",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "Uniform", "value": "uniform"},
                                    {"label": "Normal", "value": "normal"}
                        ],value="uniform",id="QuantileTransformer-output_distribution"
                        ,persistence=True,persistence_type="memory"),
                    
# =============================================================================
#                     dbc.Label("Ignore Implicit Zeros"),
#                     dbc.RadioItems(options=[
#                                     {"label": "True", "value": True},
#                                     {"label": "False", "value": False}
#                         ],value=True,id="QuantileTransformer-info-parameter-ignore_implicit_zeros",)
# =============================================================================
                    
                ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="QuantileTransformer-info-btn",n_clicks=0,),
            dbc.Alert(infoText("QuantileTransformer"),
                      id="QuantileTransformer-info-text",dismissable=True,color="info",is_open=False,),
        ]),
    id="QuantileTransformer-collapse",is_open=False,),
]



    #--------------------------
    
scalingAlgo["Robust Scaler"]={"with_centering":"","with_scaling":"",
                              "quantile_range":"","unit_variance":""}

RobustScaler_content = [
    dbc.CardHeader(getScalingHeader("Robust Scaler")),
        
    dbc.Collapse(
    dbc.CardBody(
        [ 
            dbc.FormGroup([

                    html.Div(dbc.Label("With Centering",style={"font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False}
                        ],value=True,id="Robust Scaler-with_centering"
                        ,persistence=True,persistence_type="memory"),
                    
                    html.Div(dbc.Label("With Scaling",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False}
                        ],value=True,id="Robust Scaler-with_scaling"
                        ,persistence=True,persistence_type="memory"),
                    
                    html.Div(dbc.Label("Unit Variance",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False}
                        ],value=False,id="Robust Scaler-unit_variance"
                        ,persistence=True,persistence_type="memory"),
                    
                    
                    html.Div(dbc.Label("Quantile Range",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    dcc.RangeSlider( id="Robust Scaler-quantile_range", min=0, max=100, value=[25, 75],
                                            tooltip={"always_visible": True,"placement":"bottom"},step=25,
                                            persistence=True,persistence_type="memory")
                    
                    
                ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="Robust Scaler-info-btn",n_clicks=0,),
            dbc.Alert(infoText("Robust Scaler"),
                      id="Robust Scaler-info-text",dismissable=True,color="info",is_open=False,),
        ]),
        id="Robust Scaler-collapse",is_open=False,),

]


    #--------------------------
scalingAlgo["Standard Scaler"]={"with_mean":"","with_std":""}

standard_Scaler_content = [
    dbc.CardHeader(getScalingHeader("Standard Scaler")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                    html.Div(dbc.Label("With Mean",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False}
                        ],value=True,id="Standard Scaler-with_mean"
                        ,persistence=True,persistence_type="memory"),
                    
                    html.Div(dbc.Label("With STD",style={"margin-top": "15px","margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    dbc.RadioItems(options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False}
                        ],value=False,id="Standard Scaler-with_std"
                        ,persistence=True,persistence_type="memory"),

                    ],),
            
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="Standard Scaler-info-btn",n_clicks=0,),
            dbc.Alert(infoText("Standard Scaler"),
                      id="Standard Scaler-info-text",dismissable=True,color="info",is_open=False,),
        ]),
    id="Standard Scaler-collapse",is_open=False,),

]




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Card Layouts   
scaling_content =dbc.Card([
                 dbc.CardBody([
                        dbc.Row([
                                dbc.Col(dbc.Card(maxAbs_Scaler_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(minmax_Scaler_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(normalizer_Scaler_content, color="secondary", outline=True)),
                                ],className="mb-4",),
                        
                        dbc.Row([

                                 dbc.Col(dbc.Card(powerTransformer_Scaler_content, color="secondary", outline=True)),
                                 dbc.Col(dbc.Card(quantileTransformer_Scaler_content, color="secondary", outline=True),),
                                 dbc.Col(dbc.Card(RobustScaler_content, color="secondary", outline=True),),
                                ],className="mb-4",),
                        
                        dbc.Row([
                                dbc.Col(dbc.Card(standard_Scaler_content, color="secondary", outline=True),width=4),
                                
                                ],className="mb-4",),
                            ])
                ],className="mt-3",color="dark", outline=True)




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#calllbacks of scaling info

    #--------------------------
#list of all parameters of scaling algo
all_scaling_par=[]
for key in scalingAlgo.keys():
    for par in scalingAlgo[key]:
        all_scaling_par=all_scaling_par+[key+"-"+par]
  
@app.callback(
    Output("scaling_tab_data", 'data'),
    [Input("{}-checklist-algo".format(_), 'value') for _ in scalingAlgo.keys()],
    [Input("{}".format(_), 'value') for _ in all_scaling_par],
    [State("scaling_tab_data", 'data')]
    )
def get_scale_tab_algo_input(*args):
    options=args[0:len(scalingAlgo.keys())] #all active algo
    para=args[len(scalingAlgo.keys()):-1] #all par value
    data=args[-1]   #data object
      
         
    #create a list of active algos
    algos=",".join((str(val[0]) for val in options if val))
    algos_list =[s for s in algos.split(',')]
    algos_list = list(filter(None, algos_list))
    
    if len(algos_list)>0:
        
        #dict to store active algos and respective features
        active_scalingAlgo={k:[] for k in algos_list}
        
        #assign all para to its key using scalingAlgo
        i=0
        for key in scalingAlgo.keys():
            for par in scalingAlgo[key]:
                scalingAlgo[key][par]=para[i]
                i+=1
        
        #fetch para of active algo only
        for key in active_scalingAlgo.keys():
            active_scalingAlgo[key]=scalingAlgo[key]
       
        data=active_scalingAlgo
    
    return data

    
 
    #--------------------------    
def genrateInfoCallback(id_):
    @app.callback(
    Output(id_+"-info-text", "is_open"),
    [Input(id_+"-info-btn", "n_clicks")],
    [State(id_+"-info-text", "is_open")],
    )
    
    def id__(n, is_open): 
        if n:
            return not is_open
        return is_open

# =============================================================================
# def genrateInfoCallback(id_):  
#     if id_ in allIds.keys():
#         #text for info
#         global text,link,idt
#         idt=id_
#         print(idt) 
#     
#         text=allIds[id_][0]
#         link=allIds[id_][1]
#         text=html.Label([text,html.Br() ,html.Br() ," Please refer to the ",
#                                html.A('scikit-learn user guide', href=link,target="_blank",style={"color": "black"}),
#                                " for further details."],
#                                    style={"text-align": "Justify"})
#     else:
#         text="empty"
#     
#     @app.callback(
#     [Output(id_+"-info-text", "children"),    
#     Output(id_+"-info-text", "is_open")],
#     [Input(id_+"-info-btn", "n_clicks")],
#     [State(id_+"-info-text", "is_open")],
#     ) 
#     
#     def id__(n, is_open): 
#         print("dcvx") 
#         print(text)  
#         if n:
#             return text,not is_open
#         
#         return text,is_open
#      
# =============================================================================
# =============================================================================
#     #--------------------------    
# def genrateInfoTextCallback(id):
#     #text for info
#     global text,link,idt
#     idt=id
#     print(idt) 
# 
#     text=allIds[id][0]
#     link=allIds[id][1]
#     text=html.Label([text,html.Br() ,html.Br() ," Please refer to the ",
#                            html.A('scikit-learn user guide', href=link,target="_blank",style={"color": "black"}),
#                            " for further details."],
#                                style={"text-align": "Justify"})
#     @app.callback(
#     Output(id+"-info-text", "children"),
#     [Input(id+"-info-text", "n_clicks")],
#     )
#     
#     def id_(n_clicks):
#         print(idt) 
#         print(n_clicks) 
#         if n_clicks:   
#             return text   
#         return idt
# 
# 
# =============================================================================
       
    
    #--------------------------    
def genrateAlertCallback(id):
            
    @app.callback(
    Output(id+"-alert-text", "is_open"),
    [Input(id+"-alert-btn", "n_clicks")],
    [State(id+"-alert-text", "is_open")],
    )
    
    def id(n, is_open):
        if n:  
            return not is_open
        return is_open
    
    #--------------------------    
def genrateCollapseCallback(id):
    @app.callback(
        Output(id+"-collapse", "is_open"),
        [Input(id+"-collapse-button", "n_clicks")],
        [State(id+"-collapse", "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
 

#genrateInfoTextCallback("MaxAbs Scaler")
genrateInfoCallback("MaxAbs Scaler")
genrateCollapseCallback("MaxAbs Scaler")

#genrateInfoTextCallback("MinMax Scaler")
genrateInfoCallback("MinMax Scaler")
genrateCollapseCallback("MinMax Scaler")

#genrateInfoTextCallback("Normalizer") 
genrateInfoCallback("Normalizer")
genrateCollapseCallback("Normalizer")

#genrateInfoTextCallback("PowerTransformer")
genrateInfoCallback("PowerTransformer")
genrateCollapseCallback("PowerTransformer")

#genrateInfoTextCallback("QuantileTransformer")
genrateInfoCallback("QuantileTransformer")
genrateCollapseCallback("QuantileTransformer")

#genrateInfoTextCallback("Robust Scaler")
genrateInfoCallback("Robust Scaler")
genrateCollapseCallback("Robust Scaler")

#genrateInfoTextCallback("Standard Scaler")
genrateInfoCallback("Standard Scaler")
genrateCollapseCallback("Standard Scaler")
