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
from subScript import runSubscript


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# =============================================================================
# Upload data tab
# 
# =============================================================================

upload_data_sidePanel=dbc.Card([dbc.CardBody([
                    html.Div(dbc.Label("Upload Input Data",style={"font-weight": "bold","font-size": "16px"})),
                    du.Upload(
                        id='autoML-data',
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
                                    ],value= "\t", clearable=False,style={"font-size": "14px",'color': 'black'},
                                    id="autoML_sep",persistence=True,persistence_type="memory"),
                     
                    
                    dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="autoML-info-btn",n_clicks=0,style={"margin-top": "15px"}),
                    dbc.Alert("Users should upload a .csv or .txt file where a row is a sample and a column is a feature. The first and last columns should contain the sample name and target classes, respectively. NaN values are not allowed.",
                      id="autoML-info-text",dismissable=True,color="info",is_open=False,)
            
                    ])],className="mt-3",color="dark", outline=True) 

 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
inputSidePanel=dbc.Card([dbc.CardBody([ 
                dbc.Col(html.Div(id='output_autoML',style={"margin-top": "12px"}))
                                   
                ])],className="mt-3",color="dark", outline=True) 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   
runButton=dbc.Card([dbc.CardBody([
    
                            html.Div(dbc.Label("Variance Threshold",style={"font-weight": "bold","font-size": "16px"})),    
                            html.Div(dbc.Label("Removes all features with variance lower than the given threshold.",style={"margin-top": "2px","font-size": "12px",})),
                            dbc.Input(type="number",value=1, id="varTH_automl",min=0,persistence=True,persistence_type="memory"),
                            
                            html.Div(dbc.Label("Number of Features to Select (%)",style={"margin-top": "10px","font-weight": "bold","font-size": "16px"})),    
                            dbc.Input(type="number",placeholder="in percentage",value=1, id="percentile",min=1,max=100,persistence=True,persistence_type="memory"),
                            
                            dbc.Checklist(options=[{"label": "Test Set","value": "indepTestSet1"}],
                               value=[],
                               id="keepTest",
                               inline=True,switch=True,labelStyle={"font-weight": "bold",
                                                                   "font-size": "18px"},
                               labelCheckedStyle={"color": "green"},persistence=True,persistence_type="memory",
                               style={"margin-top": "10px"}),
                            html.Div(dbc.Label("It is recommended to keep an independent test set solely for the purpose of testing the model and not for any kind of training.",style={"margin-top": "2px","font-size": "12px",})),

        
                            html.Div(dbc.Button(html.I("      Run", className="fa fa-solid fa-play-circle-o"),
                            disabled=False,color="primary",id='run_autoML', className="me-1", 
                            style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}))

            ])],className="mt-3",color="dark", outline=True)
        
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# =============================================================================
# Result section
# 
# =============================================================================

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       
plotOptions=html.Div([
                        dbc.Card([dbc.CardBody([
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
                        id="plotOptions_autoML",persistence=True,persistence_type="memory")
                        ])],className="mt-3",color="dark", outline=True),
    
                      html.Div(dbc.Button(html.I("      Download Results", className="fa fa-solid fa-download"), color="primary",id='download_autoML_res', className="me-1", 
                                 style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    dcc.Download(id="download-autoML_res")
                    ])
                        
                        
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       
scoreOptions_automl=dbc.Card([dbc.CardBody([
                        html.Div(dbc.Label("Score Type",style={"font-weight": "bold","font-size": "16px"})),
                        dcc.Dropdown(options=[
                            {"label": "Training Score", "value": "train"},
                            {"label": "Test Score", "value": "test"},
                        ],value='train', clearable=False,style={'color': 'black'},
                        id="scoreOptions_automl",persistence=True,persistence_type="memory"),
                        
# =============================================================================
#                         html.Div(dbc.Button(html.I("  Download", className="fa fa-solid fa-download"), color="primary",id='download', className="me-1", 
#                                  n_clicks=None,style={"margin-top": "15px","width":"100%","font-weight": "bold","font-size": "16px"}),
#                          className="d-grid gap-2 d-md-flex justify-content-md-end")
# =============================================================================
                        
                   ])],className="mt-3",color="dark", outline=True) 

filterOptions=dbc.Card([dbc.CardBody([
                        #dbc.Row(html.Div(dbc.Label("Subset Results",style={"font-weight": "bold","font-size": "16px"}))),
                        dbc.Row([
                            dbc.Col(html.Div(dbc.Label("Models to Show",style={"font-size": "14px"})),width=1),
                            dbc.Col(dcc.Dropdown(options=[], style={'color': 'black'},multi=True,value="all",
                            id="modelOptions_autoML",persistence=True,persistence_type="memory"),width=5),
                            
                            dbc.Col(html.Div(dbc.Label("Metrics to Show",style={"font-size": "14px"})),width=1),
                            dbc.Col(dcc.Dropdown(options=[], style={'color': 'black'},multi=True,
                            value="all",id="metricOptions_autoML",persistence=True,persistence_type="memory"),width=2),
                            
                            dbc.Col(html.Div(dbc.Label("Sort By",style={"font-size": "14px"})),width=1),
                            dbc.Col(dcc.Dropdown(options=[], style={'color': 'black'},multi=False,value="all",
                            id="sortBy_autoML",persistence=True,persistence_type="memory"),width=2)
                
                            ]),
                        dbc.Row(html.Br()),
                        dbc.Row([
                            dbc.Col(dbc.Alert("You have selected All/ Top options together or with individual models/metrics options. In such situtation all (or top) models/metrics will be used.",
                                              id="filterOptions_multiwarning_autoML",dismissable=True,is_open=False,color="secondary"))
                        ]),
                        ])],className="mt-3",color="dark", outline=True)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
spyderOptions=dbc.Card([dbc.CardBody([
                        
                        dbc.Row([
                        dbc.Col([ html.Div(dbc.Label("Color:",style={"font-weight": "bold","font-size": "14px"}))],width=1),
                        dbc.Col([
                           
                            dbc.Input(id="spyderColor_autoML", placeholder="Type a valid CSS color name or a hex code!", 
                                      type="text",persistence=True,persistence_type="memory",style={"font-size": "12px"}),   
                            
                        ],width=4),
                        
                        dbc.Col([                      
                            html.Div(dbc.Button(html.I("  Update", className="fa fa-solid fa-refresh"), color="primary",id='spyderColor_update_autoML', className="me-1", 
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
                            id={"type": "heatmapColor_autoML", "index": "myindex"} ,persistence=True,persistence_type="memory"),width=2),
                    
                        
                        
                            dbc.Col(html.Div(dbc.Label("Text:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=[
                                {"label": "True", "value": True},
                                {"label": "False", "value": False}],
                                         value=True, clearable=False,style={"font-size": "12px","color":"black"},
                            id={"type": "heatmapText_autoML", "index": "myindex"} ,persistence=True,persistence_type="memory"),width=2),
                       
                        ])

                        
                   ])],className="mt-3",color="dark", outline=True) 


barPlotOptions=dbc.Card([dbc.CardBody([
                        dbc.Row([
                        
                            dbc.Col(html.Div(dbc.Label("Color:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=list(all_palettes.keys()),
                                         value='Viridis', clearable=False,style={"font-size": "12px","color":"black"},
                            id= {"type": "barPlotColor_autoML", "index": "myindex"},persistence=True,persistence_type="memory"),width=2),
                            
                                                    
                            dbc.Col(html.Div(dbc.Label("Text:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=[
                                {"label": "True", "value": True},
                                {"label": "False", "value": False}],
                                         value=True, clearable=False,style={"font-size": "12px","color":"black"},
                            id={"type": "barPlotText_autoML", "index": "myindex"} ,persistence=True,persistence_type="memory"),width=2),
                       
                        ])

                        
                   ])],className="mt-3",color="dark", outline=True)


linePlotOptions=dbc.Card([dbc.CardBody([
                        dbc.Row([
                        
                            dbc.Col(html.Div(dbc.Label("Color:",style={"font-weight": "bold","font-size": "14px"})),width="auto"),
                            dbc.Col(dcc.Dropdown(options=list(all_palettes.keys()),
                                         value='Viridis', clearable=False,style={"font-size": "12px","color":"black"},
                            id={"type": "linePlotColor_autoML", "index": "myindex"} ,persistence=True,persistence_type="memory"),width=2),
                        ])

                        
                   ])],className="mt-3",color="dark", outline=True) 

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
resultPanel=dbc.Card([dbc.CardBody([ 
                dbc.Col(html.Div(filterOptions,style={"margin-top": "12px"})),
                dbc.Col(html.Hr(style={"background-color": "white"})),
                dbc.Col(html.Div(id='uploadResult_plots_autoML',style={"margin-top": "12px"}))                
                ])],className="mt-3",color="dark", outline=True)



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# =============================================================================
# Combine
# 
# =============================================================================
                        
import dash_loading_spinners as dls
autoML_content = dbc.Row(
    [   
        ####### side panel col
        dbc.Col([
                dbc.Row(dbc.Col(upload_data_sidePanel, width=12)),
                dbc.Row(dbc.Col(html.Div(id="hidden_scoreOptions_automl"), width=12)),
                dbc.Row(dbc.Col(html.Div(id="plotOption_side"), width=12)),
                dbc.Row(dbc.Col(html.Div(id="runButton"), width=12)),
                dbc.Row(html.Div(runButton,id="hidden2",style={'display': 'none'})),
                dbc.Row(html.Div(heatmapOptions,id="hidden3",style={'display': 'none'})),
                dbc.Row(html.Div(spyderOptions,id="hidden4",style={'display': 'none'})),                  
                dbc.Row(html.Div(barPlotOptions,id="hidden6",style={'display': 'none'})),
                dbc.Row(html.Div(linePlotOptions,id="hidden7",style={'display': 'none'})),
                dbc.Row(html.Div(scoreOptions_automl,id="hidden8",style={'display': 'none'})),
                ],width=2),

        dbc.Col(dls.Hash(inputSidePanel,color="#FFFFFF"), width=10),
        #dbc.Col(inputSidePanel, width=10),

        dbc.Row(html.Div(id="hidden_autoML",style={'display': 'none'})),

    ]
)




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# =============================================================================
# Callbacks
# 
# =============================================================================
file={}

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#Read uploaded data
@du.callback(
      Output(component_id='runButton', component_property='children'),
    id="autoML-data"
)
def getFilename_autoML(filenames): 
    global file
    if filenames!=None:
        file=filenames[0] 
        filenames=None
        return runButton
        
       
@du.callback(
      Output(component_id='hidden_autoML', component_property='children'),
    id="autoML-data"
)
def getFilename_autoML(filenames): 
    if filenames!=None:
        return spyderOptions

   
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      

results,results_NA,failedModels,trainedModels,resultBackup,selFeat_df,logFolder,testScore={},  {},{},{},{},{},"",{}
import shutil
@app.callback(
    [Output('output_autoML', 'children'),
    Output('plotOption_side', 'children'),
    Output('run_autoML', 'disabled')],
    [Input('run_autoML', 'n_clicks'),
    Input(component_id='autoML_sep', component_property='value'),
    Input('varTH_automl', 'value'),
    Input('percentile', 'value'),
    Input('keepTest', 'value')]
       
)

def runAutoML(n_clicks,sep,varTH_automl,percentile,keepTest): 
    if n_clicks is not None:
        try:
            #read file  
            global inputData,logFolder
            inputData=pd.read_csv(file,index_col=0,sep=sep)
            X= inputData.iloc[:,0:-1]
            y = inputData.iloc[:,-1]
            
            #del all folders from upload loc if there are too much
            filelist = [ f for f in os.listdir("./uploads/") if not f.startswith('.')]
            if len(filelist)>5:
                for f in filelist:
                    shutil.rmtree(os.path.join("./uploads/", f))
                
            #check for NAN values
            if inputData.isnull().values.any():
                return "Given file contains NaN values. NaN values are not allowed.","",True
            else:
                
                #del all folders from autoML_output loc if there are too much
                filelist = [ f for f in os.listdir("./autoML_output/") if not (f.startswith('.') or f.endswith('.zip'))]
                if len(filelist)>0:
                    for f in filelist:
                        shutil.rmtree(os.path.join("./autoML_output/", f))
                        
                date = datetime.now().strftime("%I_%M_%S_%p-%d_%m_%Y")
                runSubscript(inputData,date,varTH_automl,percentile,keepTest)
                
                #read data
                logFolder = os.path.join(os.getcwd(),"autoML_output/"+date)
                fileName=logFolder+'/trainedModels.pkl'
                
                global results,results_NA,failedModels,trainedModels,resultBackup,selFeat_df,testScore
                testScore={}
                with open(fileName, 'rb') as handle:
                    trainedModels=pickle.load(handle)
                    print(trainedModels)  
                    refit_Metric=trainedModels["refit_Metric"]
                    selFeat_df=getSelFeat_df(trainedModels["featSel_name"])
                    
                    if "testScore" in trainedModels.keys():
                        testScore=pd.DataFrame(trainedModels["testScore"]).T
                        testScore=changeColIndex(testScore)
                        del trainedModels["testScore"]
            
                    del trainedModels["featSel_name"]
                    del trainedModels["refit_Metric"]
                         
                    results,results_NA,failedModels=getResultDF(trainedModels,refit_Metric)
                    resultBackup=results
            
                        
                #del all folders from upload loc if there are too much
                filelist = [ f for f in os.listdir("./uploads/") if not f.startswith('.')]
                if len(filelist)>0:
                    for f in filelist:
                        shutil.rmtree(os.path.join("./uploads/", f))
                
                return resultPanel,plotOptions,True
        
        except Exception as e:       
            plot=html.Label([html.Strong("Following exception occurred:"),html.Br(),str(e)],style={"text-align": "Justify"})

            return plot,"",True
    return "","",False


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#update subset dropdown options
@app.callback(Output('modelOptions_autoML', 'options'),
              Output('metricOptions_autoML', 'options'),
              Output('modelOptions_autoML', 'value'),
              Output('metricOptions_autoML', 'value'),
              Output('modelOptions_autoML', 'multi'),
              Output('hidden_scoreOptions_automl', 'children'),
              
             [Input("plotOptions_autoML", "value")])
def updateDropdown(plotType):
    global modelOtionList,metricOtionList

    if (len(trainedModels)>0):
        #update model options and metric options
        modelOtionList_upd=getOptionList(list(results.index),"models") 
        metricOtionList_upd=getOptionList(list(results.columns),"metric")
        
        if plotType=="pipeline": 
            if isinstance(testScore, pd.DataFrame):
                return modelOtionList_upd[4:],metricOtionList_upd,modelOtionList_upd[4]["value"],"all",False,scoreOptions_automl

            else:
                return modelOtionList_upd[4:],metricOtionList_upd,modelOtionList_upd[4]["value"],"all",False,""
        else:     
            if isinstance(testScore, pd.DataFrame):
                return modelOtionList_upd,metricOtionList_upd,"all","all",True,scoreOptions_automl
            else:
                return modelOtionList_upd,metricOtionList_upd,"all","all",True,""

                
    else:
        return modelOtionList,metricOtionList,"all","all",True,""
    



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
import re
#control multiple option of ddropdown based on previous selection
@app.callback(Output('filterOptions_multiwarning_autoML', 'is_open'),
               Output('sortBy_autoML', 'options'),

             [Input('modelOptions_autoML', 'value'),
              Input('metricOptions_autoML', 'value'),
              Input('metricOptions_autoML', 'options'),

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
from helperFunctions import *

@app.callback(Output("uploadResult_plots_autoML", "children"),
              [Input("plotOptions_autoML", "value"),
              #Input(component_id='hidden2', component_property='children'),
              Input({"type": "barPlotColor_autoML", "index": ALL}, "value"),
              Input( {"type": "barPlotText_autoML", "index": ALL} , "value"),
              Input({"type": "linePlotColor_autoML", "index": ALL} , "value"),
              Input({"type": "heatmapColor_autoML", "index": ALL} , "value"),
              Input({"type": "heatmapText_autoML", "index": ALL} , "value"),
              State( "spyderColor_autoML" , "value")],
              Input(component_id='spyderColor_update_autoML', component_property='n_clicks'),
              Input("modelOptions_autoML", "value"),
              Input("metricOptions_autoML","value"),
              Input("sortBy_autoML","value"),
              Input("scoreOptions_automl","value")

                   )
def changePlot(value,
               #n_clicks,
               barPlotColor,barPlotText,
               linePlotColor,
               heatmapColor,heatmapText,spyderColor,spyderColor_update,
               models,metrics,sortBy,scoreOptions_automl):
    
    if (len(trainedModels)>0):
        
        #update result df as per user input
        if scoreOptions_automl=="train" or not isinstance(testScore, pd.DataFrame):
            results=subsetResltDF(resultBackup,models,metrics,sortBy)
        else:
            results=subsetResltDF(testScore,models,metrics,sortBy)
         
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

            fig=getHeatmap(results,heatmapColor[-1],heatmapText[-1])

            return html.Div([a,b,c,d,fig])
        
        elif value == "barPlot":    
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions)
            d=html.Div(linePlotOptions,style={"display":"none"})
 
 
            fig=getBarPlot(results,barPlotColor[-1],barPlotText[-1])

            return html.Div([a,b,c,d,fig])
        
        elif value == "linePlot":    
            a=html.Div(heatmapOptions,style={"display":"none"})
            b=html.Div(spyderOptions,style={"display":"none"})
            c=html.Div(barPlotOptions,style={"display":"none"})
            d=html.Div(linePlotOptions)
            
            fig=getLinePlot(results,linePlotColor[-1])

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
 
import zipfile
@app.callback(

    Output("download-autoML_res", "data"), 
    Input("download_autoML_res","n_clicks")

  
) 
def down_autoML_result(n_clicks):

            
    if (logFolder!="") and n_clicks:
        
        #del all file from templ loc if there are too much
        filelist = [ f for f in os.listdir("./autoML_output/") if f.endswith(".zip") ]
        if len(filelist)>2:
            for f in filelist:
                os.remove(os.path.join("./autoML_output/", f))
            
        #zip folder   
        name = logFolder
        zip_name = name + '.zip'
        
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for folder_name, subfolders, filenames in os.walk(name):
                for filename in filenames:
                    file_path = os.path.join(folder_name, filename)
                    zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))
        
        zip_ref.close()
        
        return dcc.send_file(zip_name)
     
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      
genrateInfoCallback("autoML")
genrateCollapseCallback("autoML")    



    
