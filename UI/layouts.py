#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 17:17:20 2021

@author: akshay
"""

from dash import dcc,html
import dash_bootstrap_components as dbc
from app import app
from dash.dependencies import Input, Output,State

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# =============================================================================
# preprocessing data tab
# 
# =============================================================================

#left panel preprocessing data tab
preprocessing_sidePanel = dbc.Card([dbc.CardHeader(
        dbc.Tabs(
            [
                dbc.Tab(label="Scaling", tab_id="scaling"),
                dbc.Tab(label="Data Resampling", tab_id="dataSampling"),
                dbc.Tab(label="Feature Selection", tab_id="featureSelection"),

            ],
            id="preprocessing_sidePanel",
            active_tab="scaling",
            card=True,
            
        )),
    ],className="mt-3",color="dark", outline=True)






#Right panel preprocessing data tab

    #content for scaling tab
    #--------------------------
import UI.scaling as scaling

    #content for dataSampling tab
    #--------------------------
import UI.dataSampling as dataSampling
dataSampling_content =dbc.Card([
                                 dbc.CardHeader(
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(label="Oversampling", tab_id="Oversampling"),
                                            dbc.Tab(label="Undersampling", tab_id="Undersampling"),
                                        ],id="data_resamp_tabs", active_tab="Oversampling", card=True,
                                    )
                                    ),dbc.CardBody(html.P(id="data_resamp_tabs_content", className="mt-3")),
                                ],className="mt-3",color="secondary", outline=True)

    #content for feature selection tab
    #--------------------------
import UI.featureSelection as featureSelection

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#combine panel

preprocessing_content =dbc.Row(
    [   
        dbc.Col(preprocessing_sidePanel, width=2),
        dbc.Col(html.Div(id='preprocessing-main-page-content'), width=10),
    ])





#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# =============================================================================
# Model Evaluation  tab
# 
# =============================================================================
import UI.modelEvaluation as modelEvaluation

modelEval_content =dbc.Card([
                                 dbc.CardHeader(
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(label="Methods", tab_id="methods"),
                                            dbc.Tab(label="Metrics", tab_id="metrics"),
                                        ],id="modelEval_tabs", active_tab="methods", card=True,
                                    )
                                    ),dbc.CardBody(html.P(id="modelEval_tabs_content", className="mt-3")),
                                ],className="mt-3",color="secondary", outline=True)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#calllbacks of preprocessing data tab
   

    #--------------------------       
    #callback for right panel content
@app.callback(Output("preprocessing-main-page-content", "children"), [Input("preprocessing_sidePanel", "active_tab")])
def switch_tab(at):
    if at == "scaling":
        return scaling.scaling_content
    elif at == "dataSampling":
        return dataSampling_content
    elif at == "featureSelection":
        return featureSelection.featureSelection_content
    return html.P("This shouldn't ever be displayed...")

    #--------------------------       
    #callback for Data Resampling sub tabs
@app.callback(Output("data_resamp_tabs_content", "children"), [Input("data_resamp_tabs", "active_tab")])
def switch_tab_data_resamp(at):
    if at == "Oversampling":
        return dataSampling.oversampling_content
    elif at == "Undersampling":
        return dataSampling.undersampling_content
    return html.P("This shouldn't ever be displayed...")


    #--------------------------       
    #callback for Model Evaluation sub tabs
@app.callback(Output("modelEval_tabs_content", "children"), [Input("modelEval_tabs", "active_tab")])
def switch_tab_modelEval(at):
    if at == "methods":
        return modelEvaluation.modelEval_methods
    elif at == "metrics":
        return modelEvaluation.modelEval_metrics
    return html.P("This shouldn't ever be displayed...")

if __name__ == '__main__':
    app.run_server(debug=True)
    

    
