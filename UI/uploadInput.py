#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 17:16:54 2021

@author: akshay
"""

from dash.dependencies import Input, Output,State
from app import app
import base64
import datetime
import io
import pandas as pd
import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import dash_table,html,dcc
import UI.result
from helperFunctions import *
from UI.scaling import genrateInfoCallback, genrateCollapseCallback

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# =============================================================================
# Upload data tab
# 
# =============================================================================

upload_data_sidePanel=dbc.Card([dbc.CardBody([
                    html.Div(dbc.Label("Upload Input Data",style={"font-weight": "bold","font-size": "16px"})),
                    du.Upload(
                        id='upload-data',
                        text='Drag and Drop or Select File!',
                        text_completed='Uploaded: ',
                        text_disabled='The uploader is disabled.',
                        cancel_button=True,
                        pause_button=False,
                        disabled=False,
                        filetypes=['csv','txt'],
                        chunk_size=1,
                        default_style={'lineHeight': '1','minHeight': '1',},
                        upload_id=None,
                        max_files=1,),
                    
                        html.Div(dbc.Label("Separator",style={"font-weight": "bold","font-size": "16px","margin-top": "15px"})),
                        dcc.Dropdown(
                                    options=[
                                        {"label": ",", "value": ","},
                                        {"label": "Tab", "value": "\t"},
                                        {"label": "Spcae", "value": " "},                     
                                    ],value= ",", clearable=False,style={"font-size": "14px",'color': 'black'},
                                    id="uploadInput_sep",persistence=True,persistence_type="memory"),
                     
                    
                    dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="uploadInput-info-btn",n_clicks=0,style={"margin-top": "15px"}),
                    dbc.Alert("Users should upload a .csv or .txt file where a row is a sample and a column is a feature. The first and last columns should contain the sample name and target classes, respectively. NaN values are not allowed.",
                      id="uploadInput-info-text",dismissable=True,color="info",is_open=False,)
            
                    ])],className="mt-3",color="dark", outline=True) 



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       
InputData_plotOptions=dbc.Card([dbc.CardBody([
                        html.Div(dbc.Label("Plot/Table Type",style={"font-weight": "bold","font-size": "16px"})),
                        dcc.Dropdown(options=[
                            #{"label": "Histogram", "value": "histogram"},
                            {"label": "Density Plot", "value": "densityPlot"},
                            {"label": "Scatter matrix plot", "value": "scatterMatrix"},
                            {"label": "Box plot", "value": "boxPlot"},
                            {"label": "Area plot", "value": "areaPlot"},
                            {"label": "Class Distribution", "value": "classDist"},
                            {"label": "Stats. Table", "value": "table"},


                        ],value='densityPlot', clearable=False,style={'color': 'black'},
                        id="InputData_plotOptions",persistence=True,persistence_type="memory"),
                        
                        #color
                        html.Div(dbc.Label("Plot Color",style={"font-weight": "bold","font-size": "16px","margin-top": "15px"})),
                        dcc.Dropdown(options=list(all_palettes.keys()),
                                         value='Pastel1', clearable=False,style={"font-size": "12px","color":"black"},
                            id="PlotColor",persistence=True,persistence_type="memory"),
                        
                        #features
                        html.Div(dbc.Label("Select Features",style={"font-weight": "bold","font-size": "16px","margin-top": "15px"})),
                        dcc.Dropdown(options=[], style={'color': 'black'},multi=True,value=None,
                            id="selectFeat",persistence=True,persistence_type="memory"),
                        
                        

                                                
                   ])],className="mt-3",color="dark", outline=True) 


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
denPlotOptions=dbc.Card([dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Div(dbc.Label("Curve Type",style={"font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=[
                                                {"label": "kernel density estimation (KDE)", "value": "kde"},
                                                {"label": "Normal Distribution", "value": "normal"}],
                                style={'color': 'black'},value="kde",
                            id={"type": "curveType", "index": "myindex"} ,persistence=True,persistence_type="memory"),width=5),
                   
                            ]),
                        dbc.Row(html.Br()),
                    
     ])],className="mt-3",color="dark", outline=True)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
scatterMatOptions=dbc.Card([dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Div(dbc.Label("Diagonal Plot Type",style={"font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=[
                                                {"label": "Histogram", "value": "histogram"},
                                                {"label": "Box", "value": "box"},
                                                {"label": "Scatter", "value": "scatter"}],
                                style={'color': 'black'},value="histogram",
                            id="diagPlottype" ,persistence=True,persistence_type="memory"),width=5),
                   
                            ]),
                        dbc.Row(html.Br()),
                    
     ])],className="mt-3",color="dark", outline=True)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   
inputSidePanel=dbc.Card([dbc.CardBody([ 
                dbc.Col(html.Div(id='output_data_upload',style={"margin-top": "12px"}))
                                   
                ])],className="mt-3",color="dark", outline=True) 


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   
upload_data_content = dbc.Row(
    [   
        ####### side panel col
        dbc.Col([
                dbc.Row(dbc.Col(upload_data_sidePanel, width=12)),
                dbc.Row(dbc.Col(InputData_plotOptions, width=12)),
                dbc.Row(html.Div(denPlotOptions,id="hidden1",style={'display': 'none'})),                  
                dbc.Row(html.Div(scatterMatOptions,id="hidden2",style={'display': 'none'})),                  

                ],width=2),
        dbc.Col(inputSidePanel, width=10),
        dbc.Row(html.Div(id="hidden_1",style={'display': 'none'})),

    ]
)


inputData,file={},None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#Read uploaded data
@du.callback(
     Output(component_id='hidden_1', component_property='children'),
    id="upload-data"
)
def getFilename(filenames):
    global file
               
    if filenames!=None:
        file=filenames[0]
                
        return scatterMatOptions

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  #for scatter matrix from figure factory. Check it later.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      
@app.callback(
    Output('output_data_upload', 'children'),

              [
               Input('upload-data', 'isCompleted'),
               Input(component_id='uploadInput_sep', component_property='value'),
               Input('InputData_plotOptions', 'value'),
               Input("PlotColor", "value"),
               Input("selectFeat", "value"),
               Input({"type": "curveType", "index": ALL}, "value"),
               Input("diagPlottype" , "value"),

               ]
               
               )

def update_output(is_completed,sep,InputData_plotOptions,
                  PlotColor,goi,
                  curveType,diagPlottype):
    
    #check if a file has been uploaded
    if is_completed and file is not None:
   
        try:
            #read file  
            global inputData,featoptions
            inputData=pd.read_csv(file,index_col=0,sep=sep)
                
            #check for NAN values
            if inputData.isnull().values.any():
            
                a=html.Div(scatterMatOptions,style={"display":"none"})
                b=html.Div(denPlotOptions,style={"display":"none"})
                plot="Given file contains NaN values. NaN values are not allowed."
                    
                return html.Div([a,b,plot])
            else:
                inputData.iloc[:,-1]=inputData.iloc[:,-1].astype(str)
                df=inputData.iloc[: ,0:5]   
                                
                #update result df as per user input

                pal=PlotColor
                    
                if InputData_plotOptions == "table":
                    #df=inputData[goi].describe()  
                    X= inputData.iloc[:,0:-1]

                    df=pd.DataFrame()
                    df["mean"]=X.mean()
                    df["Std.Dev"]=X.std()
                    df["Var"]=X.var()

                    
                    a=html.Div(scatterMatOptions,style={"display":"none"})
                    b=html.Div(denPlotOptions,style={"display":"none"})
                    plot=getInputDataTable(df)
                    
                    return html.Div([a,b,plot])
                
                
                elif InputData_plotOptions == "densityPlot": 
                    a=html.Div(denPlotOptions)
                    b=html.Div(scatterMatOptions,style={"display":"none"})
                    plot=getDistPlot(inputData,goi,pal,curveType[-1])
                    
                    return html.Div([a,b,plot])
                    
                      
                
                elif InputData_plotOptions == "scatterMatrix":
                    a=html.Div(scatterMatOptions)
                    b=html.Div(denPlotOptions,style={"display":"none"})
                    plot=getScatterMatrix(inputData,goi,pal,diagPlottype)
                    #plot=getDistPlot(inputData,goi,pal,curveType[-1])

                    return html.Div([a,b,plot])
               
                elif InputData_plotOptions == "boxPlot":    
                    a=html.Div(scatterMatOptions,style={"display":"none"})
                    b=html.Div(denPlotOptions,style={"display":"none"})
                    plot=html.Div(getBoxPlot(inputData,goi,pal),style={ 'width': '100%'})
                    
                    return html.Div([a,b,plot])
  
                
                elif InputData_plotOptions == "areaPlot":            
                    a=html.Div(scatterMatOptions,style={"display":"none"})
                    b=html.Div(denPlotOptions,style={"display":"none"})
                    plot=getAreaPlot(inputData,goi,pal)
                    
                    return html.Div([a,b,plot])
                       
                elif InputData_plotOptions == "classDist":            
                    a=html.Div(scatterMatOptions,style={"display":"none"})
                    b=html.Div(denPlotOptions,style={"display":"none"})
                    plot=getClasssDist(inputData,pal)
                    
                    return html.Div([a,b,plot])                
                return html.P("This shouldn't ever be displayed...")
            
            
        except Exception as e:
                        
            a=html.Div(scatterMatOptions,style={"display":"none"})
            b=html.Div(denPlotOptions,style={"display":"none"})
            #ßßplot=html.P("Following exception occurred: <br />"+str(e))
                
            plot=html.Label([html.Strong("Following exception occurred:"),html.Br(),str(e)],style={"text-align": "Justify"})
            
            return html.Div([a,b,plot])


@app.callback(
    [Output('selectFeat', 'options'),
     Output('selectFeat', 'value')],
    
    [Input('upload-data', 'isCompleted'),
     Input(component_id='uploadInput_sep', component_property='value')]
   )

def update_featOptions(is_completed,sep):
    #check if a file has been uploaded
    if is_completed and file is not None:
        try:
            #read file  
            df=pd.read_csv(file,index_col=0,sep=sep)
            
            #check for NAN values
            if df.isnull().values.any():
                return [],None
            else:
                featoptions=[{'label': i, 'value': i} for i in df.columns.tolist()]
                return featoptions,df.columns.tolist()[0:3]
            
        except Exception as e:
            return [],None
    return [],None        
    
 
     

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      
genrateInfoCallback("uploadInput")
genrateCollapseCallback("uploadInput")    