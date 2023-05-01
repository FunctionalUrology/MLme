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


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
navigation_bar = dbc.Navbar(
    dbc.Container(        [            
            #dbc.Col(dbc.NavbarBrand(html.Img(src="assets/logo.svg", height="50px")),width=2),
            dbc.Col(dbc.NavbarBrand(html.I("TOOL NAME"), style={"margin-left": "0px","font-weight": "bold","font-size": "40px","color":"white"}),width=1,align="left"),
            dbc.Col(width=9),
            html.A(dbc.Col(html.Img(src="https://www.unibe.ch/media/logo-unibern-footer@2x.png", height="80px"),align="right"),
                href="https://www.unibe.ch/index_ger.html", target="_blank",
                style={"textDecoration": "none"}),
        ],fluid=True,
    ),
   color="#444", 
   dark=True,style={"border-color": "white"}
)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
about=dbc.Card([html.Div([
    dbc.Row(html.H5("TOOL",style={"font-weight": "bold","color":"white"})),
    dbc.Row(html.P("Machine learning (ML) has become an essential tool for researchers to analyze and extract valuable insights from complex datasets. However, creating an effective ML pipeline can be a daunting task, especially for researchers who lack technical proficiency or expertise in this field. Even researchers who possess technical proficiency needs to invest significant time and effort in developing an ML pipeline. To address these challenges, we have developed a tool called ----tool name----.",style={"text-align": "justify"})),
    html.Br(),
    dbc.Row(html.P("----tool name---- empowers researchers to utilize machine learning techniques in their research, regardless of their coding and technical skills. The tool provides four primary functionalities: data exploration, auto ML, custom ML, and visualization. Users can examine their datasets and obtain initial insights through an intuitive interface with the data exploration feature. With the auto-ML feature, users can leverage a pre-built ML pipeline without requiring technical expertise. The custom ML interface is intended for advanced users and provides a user-friendly platform for developing tailor-fit ML pipelines to meet their specific research needs. Finally, Users can interpret and analyze their findings effortlessly with the help of various tables and plots using the visualization feature.",style={"text-align": "justify"})),
    html.Br(),
    dbc.Row(html.H5("Availability",style={"font-weight": "bold","color":"white"})),
    html.Div(["TOOL is developed by the ",
             html.A("Functional Urology group", href="http://www.urofun.ch/", target="_blank"),
             " at the ",
             html.A("University of Bern", href="https://www.unibe.ch/index_ger.html", target="_blank"),
             ". The source code and tutorial can be found on the ",
             html.A("TOOL GitHub repository", href="https://github.com/FunctionalUrology/", target="_blank"),
    ],style={"text-align": "justify"}),
    
    html.Br(),
    dbc.Row(html.H5("Contact",style={"font-weight": "bold","color":"white"})),
    html.P("Bug reports and new feature requests can be communicated via:"),
    html.Ul([html.Li(html.Div(["GitHub : ",html.A("https://github.com/FunctionalUrology/", href="https://github.com/FunctionalUrology/", target="_blank")]),)]),
    html.Ul([html.Li("Email : akshay.akshay@unibe.ch , ali.hashemi@dbmr.unibe.ch")]),
    html.Br(),
    dbc.Row(html.H5("Citation",style={"font-weight": "bold","color":"white"})),
    html.Div(["If TOOL helps you in any way, please cite the TOOL article:"]),
    html.Ul([html.Li(html.A("", href="", target="_blank"))]),

    ],style={"margin-left": "50px","margin-right": "50px","margin-top": "30px","font-size": "14px"})],
    style={"margin-left": "2px","margin-right": "2px","margin-top": "5px"})



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
    ],className="mt-3",color="grey", outline=False)


tabs_main = dbc.Card(
    [
     dbc.CardHeader(
        dbc.Tabs(
            [
                dbc.Tab(label="Data Exploration", tab_id="upload_data"),
                dbc.Tab(label="Auto ML", tab_id="autoML"),
                dbc.Tab(label="Custom ML", tab_id="customML"),
                dbc.Tab(label="Visualisation", tab_id="result"),
                dbc.Tab(label="About", tab_id="about")

            ],
            id="tabs_main",
            active_tab="upload_data",
            card=True,
        )),
        dbc.CardBody(html.P(id="content_main", className="mt-3")),
    ],className="mt-3",color="grey", outline=False)




app.layout = html.Div([navigation_bar,
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
    
    elif at == "about":
        return about 
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

