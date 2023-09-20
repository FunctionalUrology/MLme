#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:32:49 2022

@author: akshay
"""
from dash import dcc,html,dash_table
import dash_bootstrap_components as dbc
from app import app
from dash.dependencies import Input, Output,State
import dash_uploader as du
import pickle,dash
from helperFunctions import *
from UI.scaling import genrateInfoCallback, genrateCollapseCallback

du.configure_upload(app, 'uploads')


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
uploadResult=dbc.Card([dbc.CardBody([
                    html.Div(dbc.Label("Upload Results",style={"font-weight": "bold","font-size": "16px"})),
                    du.Upload(
                        id='uploadResult',
                        text='Drag and Drop or Select File!',
                        text_completed='Uploaded: ',
                        text_disabled='The uploader is disabled.',
                        cancel_button=True,
                        pause_button=False,
                        disabled=False,
                        filetypes=['pkl'],
                        chunk_size=50,
                        max_file_size=10240,
                        default_style={'lineHeight': '1','minHeight': '1',},
                        upload_id=None,
                        max_files=1,),
                    
                    dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="vis-info-btn",n_clicks=0,style={"margin-top": "15px"}),
                    dbc.Alert("Users should upload a 'results.pkl' file from AutoML or CustomML tab.",
                      id="vis-info-text",dismissable=True,color="info",is_open=False,)
            
                    
                    ])],className="mt-3",color="dark", outline=True) 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      
genrateInfoCallback("vis")
genrateCollapseCallback("vis")    
 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
exampleResults=dbc.Card([dbc.CardBody([
                         html.Div(dbc.Label("Example Data",style={"font-weight": "bold","font-size": "16px"})),
                         
                         dcc.Dropdown(options=[
                            {"label": "Select...", "value": "select"},
                            {"label": "Result 1", "value": "result1"},
                            {"label": "Result 2", "value": "result2"},
                            {"label": "Result 3", "value": "result3"}

                        ],value='select', clearable=False,style={'color': 'black'},
                        id="exampleResultOptions",persistence=True,persistence_type="memory"),
                         
                        
                        html.Div(dbc.Button(html.I("  Load", className="fa fa-solid fa-upload"), color="primary",id='upload_exampleResults', className="me-1", 
                                 n_clicks=None,style={"margin-top": "15px","width":"100%","font-weight": "bold","font-size": "16px"}),
                         className="d-grid gap-2 d-md-flex justify-content-md-end"),
                                               
                    ])],className="mt-3",color="dark", outline=True) 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       
plotOptions=dbc.Card([dbc.CardBody([
                        html.Div(dbc.Label("Plot/Table Type",style={"font-weight": "bold","font-size": "16px"})),
                        dcc.Dropdown(options=[
                            {"label": "Table", "value": "table"},
                            {"label": "Failed Models", "value": "failedModels"},
                            {"label": "Spyder Plot", "value": "Spyder Plot"},
                            {"label": "Heatmap", "value": "Heatmap"},
                            {"label": "Bar Plot", "value": "barPlot"},
                            {"label": "Line Plot", "value": "linePlot"},
                            {"label": "Pipeline", "value": "pipeline"},
                            {"label": "Selected Features", "value": "selFeat"},


                        ],value='table', clearable=False,style={'color': 'black'},
                        id="plotOptions",persistence=True,persistence_type="memory"),
                        
# =============================================================================
#                         html.Div(dbc.Button(html.I("  Download", className="fa fa-solid fa-download"), color="primary",id='download', className="me-1", 
#                                  n_clicks=None,style={"margin-top": "15px","width":"100%","font-weight": "bold","font-size": "16px"}),
#                          className="d-grid gap-2 d-md-flex justify-content-md-end")
# =============================================================================
                        
                   ])],className="mt-3",color="dark", outline=True) 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       
scoreOptions=dbc.Card([dbc.CardBody([
                        html.Div(dbc.Label("Score Type",style={"font-weight": "bold","font-size": "16px"})),
                        dcc.Dropdown(options=[
                            {"label": "Training Score", "value": "train"},
                            {"label": "Test Score", "value": "test"},
                        ],value='train', clearable=False,style={'color': 'black'},
                        id="scoreOptions",persistence=True,persistence_type="memory"),
                        
# =============================================================================
#                         html.Div(dbc.Button(html.I("  Download", className="fa fa-solid fa-download"), color="primary",id='download', className="me-1", 
#                                  n_clicks=None,style={"margin-top": "15px","width":"100%","font-weight": "bold","font-size": "16px"}),
#                          className="d-grid gap-2 d-md-flex justify-content-md-end")
# =============================================================================
                        
                   ])],className="mt-3",color="dark", outline=True) 



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   
 
filterOptions=dbc.Card([dbc.CardBody([
                        #dbc.Row(html.Div(dbc.Label("Subset Results",style={"font-weight": "bold","font-size": "16px"}))),
                        dbc.Row([
                            dbc.Col(html.Div(dbc.Label("Models to Show",style={"font-size": "14px"})),width=1),
                            dbc.Col(dcc.Dropdown(options=[], style={'color': 'black'},multi=True,value="all",
                            id="modelOptions",persistence=True,persistence_type="memory"),width=5),
                            
                            dbc.Col(html.Div(dbc.Label("Metrics to Show",style={"font-size": "14px"})),width=1),
                            dbc.Col(dcc.Dropdown(options=[], style={'color': 'black'},multi=True,
                            value="all",id="metricOptions",persistence=True,persistence_type="memory"),width=2),
                            
                            dbc.Col(html.Div(dbc.Label("Sort By",style={"font-size": "14px"})),width=1),
                            dbc.Col(dcc.Dropdown(options=[], style={'color': 'black'},multi=False,value="all",
                            id="sortBy",persistence=True,persistence_type="memory"),width=2)
                
                            ]),
                        dbc.Row(html.Br()),
                        dbc.Row([
                            dbc.Col(dbc.Alert("You have selected All/ Top options together or with individual models/metrics options. In such situtation all (or top) models/metrics will be used.",
                                              id="filterOptions_multiwarning",dismissable=True,is_open=False,color="secondary"))
                        ]),
                        
# =============================================================================
#                         dbc.Row(html.Br()),
#                         dbc.Row([
#                             dbc.Col(html.Div(dbc.Label("Sort By",style={"font-size": "14px"})),width=1),
#                             dbc.Col(dcc.Dropdown(options=[], style={'color': 'black'},multi=False,value="all",
#                             id="sortBy",persistence=True,persistence_type="memory"),width=5),
#                             
# # =============================================================================
# #                             dbc.Col(html.Div(dbc.Label("Metrics to Show",style={"font-size": "14px"})),width=1),
# #                             dbc.Col(dcc.Dropdown(options=[], style={'color': 'black'},multi=True,
# #                             value="all",id="metricOptions",persistence=True,persistence_type="memory"),width=3),
# # =============================================================================
#                 
#                             ])
# =============================================================================
                        
                   ])],className="mt-3",color="dark", outline=True)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
spyderOptions=dbc.Card([dbc.CardBody([
                        
                        dbc.Row([
                        dbc.Col([ html.Div(dbc.Label("Color:",style={"font-weight": "bold","font-size": "14px"}))],width=1),
                        dbc.Col([
                           
                            dbc.Input(id="spyderColor", placeholder="Type a valid CSS color name or a hex code!", 
                                      type="text",persistence=True,persistence_type="memory",style={"font-size": "12px"}),   
                            
                        ],width=4),
                        
                        dbc.Col([                      
                            html.Div(dbc.Button(html.I("  Update", className="fa fa-solid fa-refresh"), color="primary",id='spyderColor_update', className="me-1", 
                                     n_clicks=None,style={"font-size": "12px"}),
                             className="d-grid gap-2 d-md-flex justify-content-md-end")
                        ],width=1),

                        ])

                        
                   ])],className="mt-3",color="dark", outline=True) 


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
import plotly.express as px
colorscales = px.colors.named_colorscales()

heatmapOptions=dbc.Card([dbc.CardBody([
                        dbc.Row([
                        
                            dbc.Col(html.Div(dbc.Label("Color:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=colorscales,
                                         value='pubu', clearable=False,style={"font-size": "12px","color":"black"},
                            id="heatmapColor",persistence=True,persistence_type="memory"),width=2),
                    
                        
                        
                            dbc.Col(html.Div(dbc.Label("Text:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=[
                                {"label": "True", "value": True},
                                {"label": "False", "value": False}],
                                         value=True, clearable=False,style={"font-size": "12px","color":"black"},
                            id="heatmapText",persistence=True,persistence_type="memory"),width=2),
                       
                        ])

                        
                   ])],className="mt-3",color="dark", outline=True) 


barPlotOptions=dbc.Card([dbc.CardBody([
                        dbc.Row([
                        
                            dbc.Col(html.Div(dbc.Label("Color:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=list(all_palettes.keys()),
                                         value='Viridis', clearable=False,style={"font-size": "12px","color":"black"},
                            id="barPlotColor",persistence=True,persistence_type="memory"),width=2),
                            
                                                    
                            dbc.Col(html.Div(dbc.Label("Text:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=[
                                {"label": "True", "value": True},
                                {"label": "False", "value": False}],
                                         value=True, clearable=False,style={"font-size": "12px","color":"black"},
                            id="barPlotText",persistence=True,persistence_type="memory"),width=2),
                       
                        ])

                        
                   ])],className="mt-3",color="dark", outline=True)


linePlotOptions=dbc.Card([dbc.CardBody([
                        dbc.Row([
                        
                            dbc.Col(html.Div(dbc.Label("Color:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=list(all_palettes.keys()),
                                         value='Viridis', clearable=False,style={"font-size": "12px","color":"black"},
                            id="linePlotColor",persistence=True,persistence_type="memory"),width=2),
                        ])

                        
                   ])],className="mt-3",color="dark", outline=True) 

import dash_loading_spinners as dls
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
resultPanel=dbc.Card([dbc.CardBody([ 
                dbc.Col(html.Div(filterOptions,style={"margin-top": "12px"})),
                dbc.Col(html.Hr(style={"background-color": "black"})),
                dbc.Col(dls.Hash(html.Div(id='uploadResult_plots',style={"margin-top": "12px"}),size=100,color="#FFFFFF"))                
                ])],className="mt-3",color="dark", outline=True) 


uploadPanel=dbc.Col([                    
                    dbc.Row(dbc.Col(uploadResult)),
                    #dbc.Row(dbc.Col(exampleResults)),
                    dbc.Row(dbc.Col(html.Div(id="hidden_so"))),
                    dbc.Row(dbc.Col(html.Div(plotOptions,id="hidden"))),
                   
                    dbc.Row(html.Div(id="hidden2")),
                    dbc.Row(html.Div(heatmapOptions,id="hidden3",style={'display': 'none'})),
                    dbc.Row(html.Div(spyderOptions,id="hidden4",style={'display': 'none'})),                  
                    dbc.Row(html.Div(id="hidden5")),
                    dbc.Row(html.Div(barPlotOptions,id="hidden6",style={'display': 'none'})),
                    dbc.Row(html.Div(linePlotOptions,id="hidden7",style={'display': 'none'})),
                    dbc.Row(html.Div(scoreOptions,id="hidden8",style={'display': 'none'})),



                    ])

 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
uploadResult_content = dbc.Row([
                                dbc.Col(uploadPanel,width=2),
                                dbc.Col(resultPanel,width=10)
                                ])

            

    
results,results_NA,failedModels,trainedModels,resultBackup,selFeat_df,testScore={},  {},{},{},{},{},{}


import shutil
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#Read uploaded data
@du.callback(
    Output("plotOptions","value"),
    #Output("hidden5", "children")

    id="uploadResult"
)
def callback_on_completion(filenames):
    global results,results_NA,failedModels,trainedModels,resultBackup,selFeat_df,testScore
    testScore={}
    selFeat_df=pd.DataFrame()
    if filenames!=None:
        with open(filenames[0], 'rb') as handle:
             trainedModels=pickle.load(handle)

        refit_Metric=trainedModels["refit_Metric"]
        
        
        if "featSel_name" in trainedModels.keys():
            selFeat_df=getSelFeat_df(trainedModels["featSel_name"])
            del trainedModels["featSel_name"]
            
        if "testScore" in trainedModels.keys():
            testScore=pd.DataFrame(trainedModels["testScore"]).T
            testScore=changeColIndex(testScore)
            del trainedModels["testScore"]
        
        del trainedModels["refit_Metric"]
             
        results,results_NA,failedModels=getResultDF(trainedModels,refit_Metric)
        resultBackup=results
        
        #del all folders from upload loc if there are too much
        filelist = [ f for f in os.listdir("./uploads/") if not f.startswith('.')]
        if len(filelist)>5:
            for f in filelist:
                shutil.rmtree(os.path.join("./uploads/", f))

        return "table"
  
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#handle object id not found    
@app.callback(Output('hidden3', 'children'),
              Output('hidden4', 'children'),
              Output('hidden6', 'children'),
              Output('hidden7', 'children'),
              Output('hidden8', 'children'),
            [
              Input("hidden2","children"),
               Input("plotOptions", "value")])
def handleNoIDError(children,plotType):  
    return heatmapOptions,spyderOptions,barPlotOptions,linePlotOptions,scoreOptions

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#disable upload button until user select a data
@app.callback(Output('upload_exampleResults', 'disabled'),
             [Input('exampleResultOptions', 'value')])
def set_button_enabled_state(whichData):
    if whichData=="select":
        return True 
    else:
        return False
    
    
    
     
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#upload example data
@app.callback(
    Output("hidden2", "children"),
    [Input(component_id='upload_exampleResults', component_property='n_clicks'),
    Input(component_id='exampleResultOptions', component_property='value')]

)

def uploadExampleData(n_clicks,whichData):  
    global results,results_NA,failedModels,trainedModels,resultBackup,selFeat_df

    if n_clicks is not None:   
        with open('trainedModels.pkl', 'rb') as handle:
            trainedModels=pickle.load(handle)
        print(trainedModels)    
        refit_Metric=trainedModels["refit_Metric"]
        selFeat_df=getSelFeat_df(trainedModels["featSel_name"])
        
        del trainedModels["featSel_name"]
        del trainedModels["refit_Metric"]
        
        results,results_NA,failedModels=getResultDF(trainedModels,refit_Metric)
        resultBackup=results

        return "d"
    
    
    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#update subset dropdown options
@app.callback(Output('modelOptions', 'options'),
              Output('metricOptions', 'options'),
              Output('modelOptions', 'value'),
              Output('metricOptions', 'value'),
              Output('modelOptions', 'multi'),
              Output('hidden_so', 'children'),
              
             [
                 #Input("hidden5", "children"),
              Input("hidden2","children"),
               Input("plotOptions", "value")])
def updateDropdown(children,plotType):
    global modelOtionList,metricOtionList

    if (len(trainedModels)>0):
        #update model options and metric options
        modelOtionList_upd=getOptionList(list(results.index),"models") 
        metricOtionList_upd=getOptionList(list(results.columns),"metric")

        if plotType=="pipeline": 
            if isinstance(testScore, pd.DataFrame):
                return modelOtionList_upd[4:],metricOtionList_upd,modelOtionList_upd[4]["value"],"all",False,scoreOptions
            else:
                return modelOtionList_upd[4:],metricOtionList_upd,modelOtionList_upd[4]["value"],"all",False,""

        else:     
            if isinstance(testScore, pd.DataFrame):
                return modelOtionList_upd,metricOtionList_upd,"all","all",True,scoreOptions
            else:
                return modelOtionList_upd,metricOtionList_upd,"all","all",True,""
                
    else:
        return modelOtionList,metricOtionList,"all","all",True,""
    



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
import re
#control multiple option of ddropdown based on previous selection
@app.callback(Output('filterOptions_multiwarning', 'is_open'),
               Output('sortBy', 'options'),

             [Input('modelOptions', 'value'),
              Input('metricOptions', 'value'),
              Input('metricOptions', 'options'),

              ])

def updateDropdownMultiOption(rows,cols,metricOptions):
    
    #chekc for top and all with other metric and model name
    if isinstance(rows, list) and (len(rows)>1):
        #check for top
        topLen=[x for x in rows if re.search('top', x)]
        
        if ("all" in rows) or (topLen!=[]):
            return True,[]

    if isinstance(cols, list) and (len(cols)>1) and ("all" in cols):
        return True,[]
    
    #sorting
    if cols=="all" or ("all" in cols):
        #metricOptions=metricOptions.remove("all")
        
        return False,metricOptions[1:]

    else:
        metrics=[{'label': i, 'value': i} for i in cols]
        return False,metrics

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#update plots/tables as per user input
from matplotlib.colors import is_color_like
@app.callback(Output("uploadResult_plots", "children"),
              [Input("plotOptions", "value"),
              Input(component_id='hidden2', component_property='children'),
              Input("barPlotColor", "value"),
              Input("barPlotText", "value"),
              Input("linePlotColor", "value"),
              Input("heatmapColor", "value"),
              Input("heatmapText", "value"),
              State("spyderColor", "value")],
              Input(component_id='spyderColor_update', component_property='n_clicks'),
              Input("modelOptions", "value"),
              Input("metricOptions","value"),
              Input("sortBy","value"),
              Input("scoreOptions","value")

                   )
def changePlot(value,
               n_clicks,
               barPlotColor,barPlotText,
               linePlotColor,
               heatmapColor,heatmapText,spyderColor,spyderColor_update,
               models,metrics,sortBy,scoreOptions):
    
    
    if (len(trainedModels)>0) or (n_clicks is not None):
  
        if scoreOptions=="train" or not isinstance(testScore, pd.DataFrame):
            #update result df as per user input
            results=subsetResltDF(resultBackup,models,metrics,sortBy)
        else:
            results=subsetResltDF(testScore,models,metrics,sortBy)
        
        results=results.astype(float).round(4)
            
        if value == "table":
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions,style={"display":"none"})
            d=html.Div(linePlotOptions,style={"display":"none"})

            return  html.Div([a,b,c,d,getTable(results) ])  
        
        elif value == "failedModels": 
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions,style={"display":"none"})
            d=html.Div(linePlotOptions,style={"display":"none"})

            return  html.Div([a,b,c,d,getTable(failedModels) ])
        
        elif value == "Spyder Plot":
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions)
            c=html.Div(barPlotOptions,style={"display":"none"})
            d=html.Div(linePlotOptions,style={"display":"none"})

 
            if spyderColor_update is not None: 
                if(is_color_like(spyderColor)!=True):
                
                    fig=html.Div([
                        html.P("Given color value is invalid! Using default"),
                        getSpyderPlot(results,"#B9E4E8")
                        ])
                else:
                     fig=getSpyderPlot(results,spyderColor)
                    
            else:
                fig=getSpyderPlot(results,"#B9E4E8")
                
            return  html.Div([a,b,c,d,fig])
        
        elif value == "Heatmap":    
            a=html.Div(heatmapOptions)
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions,style={"display":"none"})
            d=html.Div(linePlotOptions,style={"display":"none"})

            fig=getHeatmap(results,heatmapColor,heatmapText)

            return html.Div([a,b,c,d,fig])
        
        elif value == "barPlot":    
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions)
            d=html.Div(linePlotOptions,style={"display":"none"})

            fig=getBarPlot(results,barPlotColor,barPlotText)

            return html.Div([a,b,c,d,fig])
        
        elif value == "linePlot":    
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions,style={"display":"none"})
            d=html.Div(linePlotOptions)
            
            fig=getLinePlot(results,linePlotColor)

            return html.Div([a,b,c,d,fig])
        
        elif value == "pipeline":    
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions,style={"display":"none"})
            d=html.Div(linePlotOptions,style={"display":"none"})
            fig=getPieline(trainedModels,models,"diagram-")
            

            return html.Div([a,b,c,d,fig])
        
        elif value == "selFeat":    
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions,style={"display":"none"})
            d=html.Div(linePlotOptions,style={"display":"none"})
            
            
            fig=getSelFeat_Table(selFeat_df)
            

            return html.Div([a,b,c,d,fig])
        
        return html.P("This shouldn't ever be displayed...")
   
