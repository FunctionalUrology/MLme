#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:00:26 2021

@author: akshay
"""
import dash
from dash import dcc
from app import app
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output,State,ALL
from dash.exceptions import PreventUpdate
from UI.layouts import preprocessing_content,modelEval_content
from helperFunctions import saveUserInputData
from UI.uploadInput import upload_data_content



seed_core=[
    
    dbc.CardBody(
        [
            
            html.Div(dbc.Label("Random Seed",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("A random seed is used to ensure that results are reproducible.",style={"margin-top": "12px","font-size": "12px",})),
            dbc.Input(type="number",placeholder="default 123", min=0,step=1,id="random_seed",persistence=True,persistence_type="memory"),

            html.Div(dbc.Label("No. of CPU/Core",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            dbc.Input(type="number",placeholder="default 1", min=-1,step=1,id="n_jobs",persistence=True,persistence_type="memory"),   

        html.Div(dbc.Button(html.I("  Submit", className="fa fa-solid fa-play-circle-o"), color="primary",id='generateScript', className="me-1", 
                                 style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                 className="d-grid gap-2 d-md-flex justify-content-md-end",),
        dcc.Download(id="download-data"),
       
        dbc.Alert("Hello! I am an alert",id="alert-fade",dismissable=True,is_open=True),
        html.Div(id='body-div_1'),
        ]),   
   
    ]


# =============================================================================
# buttons=[
#     
#     dbc.CardBody(
#         [
#              html.Div(dbc.Button("Generate Script", color="primary",id='generateScript', active=True, className="me-1", 
#                                  style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
#              html.Div(dcc.Download("Run", color="primary",id='runScript', active=True, className="me-1",
#                                   style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
#             
#             html.Div(id='body-div_1'),
#         ]),   
#    
#     ]
# =============================================================================


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Card Layouts   
submit_con =dbc.Card([
                 dbc.CardBody([
                        dbc.Row([
                                dbc.Col(dbc.Card(seed_core, color="secondary", outline=True),width=6),
                                #dbc.Col(dbc.Card(buttons, color="secondary", outline=True),width=6),

                                ],className="mb-4",),

                            ]), ],className="mt-3",color="dark", outline=True)

#!!!!!!
filename=""
@app.callback(
    #Output(component_id='body-div_1', component_property='children'),
    Output("alert-fade", "is_open"),
    Output("alert-fade", "children"),
    Output("alert-fade", "color"),
    
    Input(component_id='generateScript', component_property='n_clicks'),
    Input('random_seed', "value"),
    Input('n_jobs', "value"),

    [State("scaling_tab_data", 'data')],
    [State("overSamp_tab_para", 'data')],
    [State("underSamp_tab_para", 'data')],
    [State("featSel_tab_para", 'data')],
    [State("classification_tab_para", 'data')],
    [State("modelEval_tab_para", 'data')],
    [State("modelEval_metrices_tab_para", 'data')],
    [State("refit_Metric", 'data')],
    [State("indepTestSet", 'data')],    
)

def submit_data(*args): 
    n_clicks=args[0]
    data={}
    data["random_seed"]=args[1]
    data["n_jobs"]=args[2]

    data["scaling_tab_data"]=args[3]
    data["overSamp_tab_para"]=args[4]
    data["underSamp_tab_para"]=args[5]
    data["featSel_tab_para"]=args[6]
    data["classification_tab_para"]=args[7]
    data["modelEval_tab_para"]=args[8]
    data["modelEval_metrices_tab_para"]=args[9]
    data["refit_Metric"]=args[10]
    data["indepTestSet"]=args[11]
    
    if n_clicks is None: 
        return False,"cfv","danger"
    
    else:
# =============================================================================
#         with open('myfile_test.txt', 'w') as f:
#             print(data, file=f)
# =============================================================================
       #saveUserInputData(data)
        if(data["classification_tab_para"]=={}):
            return True ,"Please select ateleast one classification algorithm from Classification Algorithms tab.","danger"

        if(data["modelEval_tab_para"]=={}):
            return True ,"Please select ateleast one model evaluation method from Model Evaluation tab (besides Independent Test Set option).","danger"        
        
        if(data["modelEval_metrices_tab_para"]=={} or 
           data["modelEval_metrices_tab_para"][0]==[]):
            return True ,"Please select ateleast one evaluation metrics from Model Evaluation tab.","danger"

        else:
           global filename
           filename=saveUserInputData(data)
           return True ,"Done.","success"



@app.callback(

    Output("download-data", "data"), 
    Input("alert-fade","color"),
    Input("generateScript","n_clicks")

  
) 

def down_data(color,n_clicks):

    if (filename!="") and (color=="success") and n_clicks:
        return dcc.send_file(filename)



