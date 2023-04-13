#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 17:17:42 2021

@author: akshay
"""

from dash import dcc,html
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import dash_table

from app import app
from UI.layouts import preprocessing_content,modelEval_content
from UI.scaling import scalingAlgo
from UI.classifcationAlgo import classAlgo_content
from UI.submit import submit_con
from UI.result import uploadResult_content
from UI.uploadInput import upload_data_content
from UI.autoML import autoML_content




app.css.config.serve_locally = False

 
tabs = dbc.Card(
    [
     dbc.CardHeader(
        dbc.Tabs(
            [
                #dbc.Tab(label="Data Exploration", tab_id="upload_data"),
                dbc.Tab(label="Preprocessing", tab_id="preprocessing"),
                dbc.Tab(label="Classification Algorithms", tab_id="classAlgo"),
                dbc.Tab(label="Model Evaluation ", tab_id="modelEval"),
                dbc.Tab(label="Submit", tab_id="submit"),
                
            ],
            id="tabs",
            active_tab="preprocessing",
            card=True,
        )),
        dbc.CardBody(html.P(id="content", className="mt-3")),
    ],className="mt-3",color="secondary", outline=False)


tabs_main = dbc.Card(
    [
     dbc.CardHeader(
        dbc.Tabs(
            [
                dbc.Tab(label="Data Exploration", tab_id="upload_data"),
                dbc.Tab(label="Auto ML", tab_id="autoML"),
                dbc.Tab(label="Custom ML", tab_id="customML"),
                dbc.Tab(label="Visualisation", tab_id="result")
            ],
            id="tabs_main",
            active_tab="upload_data",
            card=True,
        )),
        dbc.CardBody(html.P(id="content_main", className="mt-3")),
    ],className="mt-3",color="secondary", outline=False)




app.layout = html.Div([
    dcc.Store(id='scaling_tab_data',data={}),
    dcc.Store(id='overSamp_tab_para',data={}),
    dcc.Store(id='underSamp_tab_para',data={}),
    dcc.Store(id='featSel_tab_para',data={}),
    dcc.Store(id='classification_tab_para',data={}),
    dcc.Store(id='modelEval_tab_para',data={}),
    dcc.Store(id='modelEval_metrices_tab_para',data={}),
    dcc.Store(id='refit_Metric',data={}), 
    dcc.Store(id='indepTestSet',data={}),
    tabs_main
    ]) 

#def getActiveAlgo(algoList):
@app.callback(
       Output("btn", "children"),
    Input("MaxAbs Scaler-collapse-button", "n_clicks")
)
def toggle_collapse(n): 
    # print(html.P("d"))
    return n

@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    #if at == "upload_data":
        #return upload_data_content
    if at == "preprocessing":
        return preprocessing_content 
    elif at == "classAlgo":
        return classAlgo_content
    elif at == "modelEval":
        return modelEval_content
    elif at == "submit":
        return submit_con

    return html.P("This shouldn't ever be displayed...")

@app.callback(Output("content_main", "children"), [Input("tabs_main", "active_tab")])
def switch_tab(at):
    if at == "upload_data":
        return upload_data_content
    elif at == "autoML":
        return autoML_content 
    elif at == "customML":
        return tabs
    elif at == "result":
        return uploadResult_content
    return html.P("This shouldn't ever be displayed...")


import webbrowser as web
web.open_new('http://127.0.0.1:8050/')

# =============================================================================
# if __name__ == '__main__':
#     app.run_server(host='127.0.0.1',debug=True,dev_tools_hot_reload=False)
# =============================================================================

if __name__ == '__main__':
    app.run_server(host='127.0.0.1',debug=False,dev_tools_hot_reload=False)    


server = app.server

