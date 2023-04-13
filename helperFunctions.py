#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:46:15 2022

@author: akshay
"""
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def getAlgoNames(comIDS):
    
    comIDS=comIDS.split(",")
    comIDS = [sub[1 : ] for sub in comIDS]
    
    #get all the algo names
    global algoName
    algoName=[]
    for item in comIDS:
        if "-" not in item and "_" not in item:
            algoName.append(item)
    return algoName

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def removeModelId(model_id,paraList):
   return {key.replace(model_id+"-", ""): value for key, value in paraList.items()}

              
                    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def getActiveAlgo(userInputData,tabNAME,models,rs,classification_Com_IDS):
    #Check if there is any algo has been selected bs user.
    #Execute only if there is atleast one algo has been selected.
    if tabNAME in userInputData.keys():
        
        
        tab_para=userInputData[tabNAME]
        
        models_ids=getAlgoNames(classification_Com_IDS)
        
        #save all actie models
        models_active={}
        i=0

        
        for model_id in models_ids:
            
            #check if a algo is active by looking at the user input data
            if model_id in tab_para.keys():
        
                model=models[i]

                #set random state for the algorithm
                if ("random_state") in list(model.get_params().keys()):
                    model.set_params(random_state=rs)
                
                ##set no of cpu's for the algorithm only if it is not None(default)
                if ("n_jobs") in list(model.get_params().keys()) \
                                      and userInputData["n_jobs"]!=None:
                    model.set_params(n_jobs=userInputData["n_jobs"])
        
                #set other parameters of respective algorithms
                paraList=tab_para[model_id]
                if len(paraList)!=0:

                    #remove model id prefix from para name
                    paraList=removeModelId(model_id,paraList)
                    
                    #remove parameters without any Input from User
                    #parameters containing None values
                    for k, v in list(paraList.items()):
                        if v == None:
                            del paraList[k]

                    #convert rangslider input (list by default) into tuple
                    #make sure it wont break remaing code ^^^^^^^^^^
                    for k, v in list(paraList.items()):
                        if type(v) == list:
                            paraList[k]=tuple(paraList[k])
                    
                    #^^^ Specially for ADASYN
                    #^^^ we need to change universal k_neighbors to n_neighbors. 
                    if model_id=="ADASYN":
                        paraList["n_neighbors"] = paraList.pop("k_neighbors")
                        
                    #^^^ Specially for RandomOverSampler
                    #^^^ since n_neighbors parameter is not required 
                    #for random oversampler, delete it 
                    if model_id=="RandomOverSampler":
                        paraList.pop("k_neighbors", None)
                        #del  paraList["k_neighbors"]
                        
                    model.set_params(**paraList)
    
                    
                models_active[model_id]=model
            i+=1
    return models_active


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from sklearn.model_selection import *

def getMoedlEvalActive(userInputData,tabNAME,modelEval_Com_IDS,rs):
    tab_para=userInputData[tabNAME]

    models_ids=getAlgoNames(modelEval_Com_IDS)
    
    #save all actie models
    models_active={}
    
    for model_id in models_ids:
            
            #check if a algo is active by looking at the user input data
            if model_id in tab_para.keys():
        
                
                 #set other parameters of respective algorithms
                paraList=tab_para[model_id]
                if len(paraList)!=0:
    
                    #remove model id prefix from para name
                    paraList=removeModelId(model_id,paraList)
                    
                    #ValueError: Setting a random_state has no effect since shuffle is False.
                    #You should leave random_state to its default (None), or set shuffle=True.
                    if "Leave" not in model_id and "shuffle" in paraList.keys():
                        if paraList["shuffle"]==False:
                            rs=None
                        
                        
                    #remove parameters without any Input from User
                    #parameters containing None values
                    for k, v in list(paraList.items()):
                        if v == None:
                            del paraList[k]
    
                    #convert rangslider input (list by default) into tuple
                    #make sure it wont break remaing code ^^^^^^^^^^
                    for k, v in list(paraList.items()):
                        if type(v) == list:
                            paraList[k]=tuple(paraList[k])
                        
                      
                    #intialize model
                    if model_id=="KFold":
                        model=KFold(**paraList,random_state=rs)
                        
                    elif model_id=="StratifiedKFold":
                        model=StratifiedKFold(**paraList,random_state=rs)
                        
                    elif model_id=="RepeatedKFold":
                        
                        model=RepeatedKFold(**paraList,random_state=rs)                    
                        
                    elif model_id=="RepeatedStratifiedKFold":
                        model=RepeatedStratifiedKFold(**paraList,random_state=rs)                    
                        
                    elif model_id=="LeaveOneOut":
                        model=LeaveOneOut(**paraList)
                        
                    elif model_id=="LeavePOut":
                        model=LeavePOut(**paraList)
    
                    elif model_id=="ShuffleSplit":
                        model=ShuffleSplit(**paraList,random_state=rs)
    
                    elif model_id=="StratifiedShuffleSplit":
                        model=StratifiedShuffleSplit(**paraList,random_state=rs)
                    
                    elif model_id=="NestedCV":
                        model=StratifiedKFold(**paraList,random_state=rs)
                          
                        
                    else:
                        print(model_id)
                        print(paraList)
    
                        print("Something is wrong")
                        break
                    
                else:
                     #intialize model without para list
                    if model_id=="KFold":
                        model=KFold(random_state=rs)
                        
                    elif model_id=="StratifiedKFold":
                        model=StratifiedKFold(random_state=rs)
                        
                    elif model_id=="RepeatedKFold":
                        model=RepeatedKFold(random_state=rs)                    
                        
                    elif model_id=="RepeatedStratifiedKFold":
                        model=RepeatedStratifiedKFold(random_state=rs)                    
                        
                    elif model_id=="LeaveOneOut":
                        model=LeaveOneOut()
    
                    elif model_id=="LeavePOut":
                        model=LeavePOut(p=10) 
    
    
                    elif model_id=="ShuffleSplit":
                        model=ShuffleSplit(random_state=rs)
    
                    elif model_id=="StratifiedShuffleSplit":
                        model=StratifiedShuffleSplit(random_state=rs)
                    
                    elif model_id=="NestedCV":
                        model=StratifiedKFold(random_state=rs)
                        
                    else:
                        print("Something is wrong 2")
                        break
 
                print(model)  

                models_active[model_id]=model
                
    return models_active

from sklearn.feature_selection import f_classif,chi2
featSel_scoreFun={"f_classif":f_classif,"chi2":chi2}

def getActiveAlgoFeatSel(userInputData,tabNAME,models,rs,classification_Com_IDS,featSel_est):
    #Check if there is any algo has been selected bs user.
    #Execute only if there is atleast one algo has been selected.
    if tabNAME in userInputData.keys():
        
        
        tab_para=userInputData[tabNAME]
        
        models_ids=getAlgoNames(classification_Com_IDS)
 
        #save all actie models
        models_active={}
        i=0
            
        for model_id in models_ids:
            
      
            #check if a algo is active by looking at the user input data
            if model_id in tab_para.keys():
                model=models[i]
                
                #set random state for the algorithm
                if ("random_state") in list(model.get_params().keys()):
                    model.set_params(random_state=rs)
                
                ##set no of cpu's for the algorithm only if it is not None(default)
                if ("n_jobs") in list(model.get_params().keys()) \
                                      and userInputData["n_jobs"]!=None:
                    model.set_params(n_jobs=userInputData["n_jobs"])
        
                #set other parameters of respective algorithms
                paraList=tab_para[model_id]
                if len(paraList)!=0:

                    #remove model id prefix from para name
                    paraList=removeModelId(model_id,paraList)
                    
                    #remove parameters without any Input from User
                    #parameters containing None values
                    for k, v in list(paraList.items()):
                        if v == None:
                            del paraList[k]

                    #convert rangslider input (list by default) into tuple
                    #make sure it wont break remaing code ^^^^^^^^^^
                    for k, v in list(paraList.items()):
                        if type(v) == list:
                            paraList[k]=tuple(paraList[k])
            
                    #change estimator name with actual estimator in parameter list
                    if "estimator" in paraList.keys():
                        estimatorname=paraList["estimator"]
                        paraList["estimator"]=featSel_est[estimatorname]
                    
                    if "score_func" in paraList.keys():
                        estimatorname=paraList["score_func"]
                        paraList["score_func"]=featSel_scoreFun[estimatorname] 
                         
                    model.set_params(**paraList)
                  

                    
                models_active[model_id]=model
            i+=1
    return models_active

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from UI.componentIDs import classification_Com_IDS,classification_models,\
    undersampling_Com_IDS,underSamp_models,\
       overrsampling_Com_IDS, overSamp_models, \
           modelEval_Com_IDS,\
               scaling_Com_IDS,scaling_models, \
                   featSel_Com_IDS, featSel_models,featSel_est
                   
import pickle
import numpy as np
from zipfile import ZipFile
import os
from datetime import datetime

def saveUserInputData(userInputData):

    #get random state4  
    if("random_seed" in userInputData.keys()):
        rs=userInputData["random_seed"]
    else:
        rs=12345
    
    #set numpy random seed
    np.random.seed(rs)
    
        
                
    scaling_tab_active=getActiveAlgo(userInputData,"scaling_tab_data",
                                       scaling_models,rs,scaling_Com_IDS)
                
    underSamp_tab_active=getActiveAlgo(userInputData,"underSamp_tab_para",
                                       underSamp_models,rs,undersampling_Com_IDS)
    
    overSamp_tab_active=getActiveAlgo(userInputData,"overSamp_tab_para",
                                       overSamp_models,rs,overrsampling_Com_IDS)    

    featSel_tab_active=getActiveAlgoFeatSel(userInputData,"featSel_tab_para",
                                       featSel_models,rs,featSel_Com_IDS,featSel_est)  
        
    classification_tab_active=getActiveAlgo(userInputData,"classification_tab_para",
                                            classification_models,rs,classification_Com_IDS)
    
    
    modelEval_tab_active=getMoedlEvalActive(userInputData,"modelEval_tab_para",
                                            modelEval_Com_IDS,rs)
    
    
    userInputData={"random_state":rs,"n_jobs":userInputData["n_jobs"],"refit_Metric":userInputData["refit_Metric"],\
          "scaling_tab_active":scaling_tab_active,"underSamp_tab_active":underSamp_tab_active,\
          "overSamp_tab_active":overSamp_tab_active,"classification_tab_active":classification_tab_active,
          "featSel_tab_active":featSel_tab_active,\
          "modelEval_tab_active":modelEval_tab_active,\
          "indepTestSet":userInputData["indepTestSet"],\
          "modelEval_metrices":userInputData["modelEval_metrices_tab_para"][0]
    }
    #return userInputData
    
    #temp location
    folder="./userInputData/"
    
    #del all file from templ loc if there are too much
    filelist = [ f for f in os.listdir(folder) if f.endswith(".zip") ]
    if len(filelist)>2:
        for f in filelist:
            os.remove(os.path.join(folder, f))
    
    #create filenames
    current_time = datetime.now().strftime("%H_%M_%S")
    fileName=folder+"inputParameters_"+current_time+".pkl"
    zipfileName=folder+"data_"+current_time+".zip"
    
    #Save user input data as pkl object
    with open(fileName, 'wb') as handle:
        pickle.dump(userInputData, handle)
    
    #zipped them
    with ZipFile(zipfileName, 'w') as zipObj2:
       # Add multiple files to the zip
       zipObj2.write(fileName)
       zipObj2.write("./scriptTemplate.py")
       zipObj2.write("./README.txt")

       #delete pkl file
       if os.path.exists(fileName):
           os.remove(fileName)
   

    return zipfileName

# =============================================================================
#     with open('userInputData_test.pkl', 'wb') as handle:
#         pickle.dump(userInputData, handle)
# =============================================================================
        

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#dash table styling
from dash import Dash, dash_table, html
import pandas as pd
from collections import OrderedDict

def discrete_background_color_bins(df, n_bins=5, columns='all'):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
        color = 'black' if i > len(bounds) / 2. else 'black'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# get dash table
import pandas
def getTable(df):
    modelName=pandas.DataFrame({"Algorithm-Evaluation Method":df.index},index=df.index)
    result = pandas.concat([modelName, df], axis=1)
    df=result  

    (styles, legend) = discrete_background_color_bins(df)                     

    return html.Div([
                html.Div(legend, style={'float': 'right'}),
                
                dash_table.DataTable(
                id='datatable-paging',
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in sorted(df.columns)],
                
                style_data={'color': 'white','backgroundColor': '#444'},
                style_header={'backgroundColor': 'black','color': 'white','fontWeight': 'bold'},
                style_table={'overflow': 'scroll','minWidth': '100%'},
                #fixed_columns={ 'headers': True, 'data': 1 },
                style_cell={'textAlign': 'left'},
                sort_action='native',
               style_data_conditional=styles ,
               export_format='xlsx',
               export_headers='display',

            )
])

def getInputDataTable(df):
    
    modelName=pandas.DataFrame({"Index":df.index},index=df.index)
    result = pandas.concat([modelName, df], axis=1)
    df=result  
    
    (styles, legend) = discrete_background_color_bins(df)                     

    return html.Div([
                html.Div(legend, style={'float': 'right'}),
                
                dash_table.DataTable(
                id='datatable-paging',
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                
                style_data={'color': 'white','backgroundColor': '#444'},
                style_header={'backgroundColor': 'black','color': 'white','fontWeight': 'bold'},
                style_table={'overflow': 'scroll','minWidth': '100%'},
                #fixed_columns={ 'headers': True, 'data': 1 },
                style_cell={'textAlign': 'left'},
                sort_action='native',
               style_data_conditional=styles ,
               export_format='xlsx',
               export_headers='display',

            )
])

def getSelFeat_Table(df):
    modelName=pandas.DataFrame({"Algorithm-Evaluation Method":df.index},index=df.index)
    result = pandas.concat([modelName, df], axis=1)
    df=result  
    
    #convert list to string 
    for index, row in df.iterrows():
        if str(row["Selected Features"]):
            continue  
        else:            
            df.loc[index,"Selected Features"]=','.join(str(e) for e in row["Selected Features"])
    
    df.dropna(inplace=True)
    
    return html.Div([
                
                dash_table.DataTable(
                id='datatable-paging',
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in sorted(df.columns)],
                
                style_data={'color': 'white','backgroundColor': '#444'},
                style_header={'backgroundColor': 'black','color': 'white','fontWeight': 'bold'},
                style_table={'overflow': 'scroll','minWidth': '100%'},
                #fixed_columns={ 'headers': True, 'data': 1 },
                style_cell={'textAlign': 'left'},
                sort_action='native',
               export_format='xlsx',
               export_headers='display',

            )
])



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def getSelFeat_df(featureIndex_name):
    selFeat_df = pd.DataFrame(index=list(featureIndex_name.keys()),columns =[ "Selected Features"]) 
    for key in featureIndex_name.keys():
        features=featureIndex_name[key]
        features=','.join(str(e) for e in features)
        selFeat_df.loc[key,"Selected Features"]=features
    return selFeat_df

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc

import plotly.io as pio
pio.renderers.default = 'browser'

config={'displaylogo': False,
           'toImageButtonOptions': {'format': 'png','scale':5}}
   

def getSpyderPlot(df_sorted,line_color):
    
    rows=int(df_sorted.shape[0]/3)+1
    cols=3
    width=1000
    height=rows*300
    title_font=14
    marker_size=4
    label_size=10
    tick_size=10
    #line_color="#B9E4E8"

    specs=[]
    for row in range(1,rows+1):
        a=[]
        for col in range(1,cols+1):
            a.append({"type": "polar"})
        specs.append(a)
            
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[i.replace('__', '-') for i in df_sorted.index.tolist()],
                       specs=specs)
    
    row=1     
    col=1
    for model in df_sorted.index:
        name=[]
        value=[]
    
        model_score=df_sorted.loc[model]
        for score in model_score.index:
            if score=="model":
                continue
            name.append(score)
            value.append(model_score.loc[score]*100)
        
        fig_tem=go.Scatterpolar(r=value,name=model,dtheta=20,
                              theta=name,fill='toself',
                              line_color=line_color)     
        fig.add_trace(fig_tem,
                  row=row, col=col)

        
        if col ==cols:
            col=1
            row+=1
        else:
            col+=1
        
    
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=title_font,family="Arial",color='black')
        i['borderpad'] =15

    fig.update_layout(font_size =label_size,template="plotly_white",
                      font_family="Arial",
                      width=width,height=height,
                      showlegend=False,margin=dict(t=50, b=50, r=50, l=50,))


    fig.update_polars(radialaxis=dict(
                          visible=True,nticks=7,
                          angle=1,
                          range=[30, 100],
                          tickfont=dict(size=tick_size)
                        ),
        angularaxis = dict(showticklabels=False, ticks='', linewidth = 0.2,showline=True,linecolor='black'))
    
    #fig.update_traces(marker=dict(size=6,line_color="black",color=px.colors.qualitative.Set1), selector=dict(type='scatterpolar'))
    fig.update_traces(marker=dict(size=marker_size,line_color="black",color=px.colors.sequential.Viridis), selector=dict(type='scatterpolar'))

    fig.update_polars(angularaxis = dict(showticklabels=True))
    
    fig.update_layout(
                 dragmode='drawopenpath',
                 newshape_line_color='#B32900',
                modebar_add=['drawline',
                'drawopenpath',
                'drawclosedpath',
                'drawcircle',
                'drawrect',
                'eraseshape']
            )
   
    return  dcc.Graph(figure=fig,config=config)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
def getHeatmap(df_sorted,heatmapColor,heatmapText):
  
  #set plot height
  if df_sorted.shape[0]<5:
      height=500
  else:
      height=int(df_sorted.shape[0]/2)*150
 
  fig = px.imshow(df_sorted,text_auto=heatmapText,color_continuous_scale=heatmapColor, aspect="auto")
  fig.update_xaxes(side="top",tickangle = 90)
  

  
  fig.update_layout( height=   height    ,       
                     dragmode='drawopenpath',
                     newshape_line_color='#B32900',
                    modebar_add=['drawline',
                    'drawopenpath',
                    'drawclosedpath',
                    'drawcircle',
                    'drawrect',
                    'eraseshape']
                )
  
   
  return dcc.Graph(figure=fig,config=config)

from bokeh.palettes import all_palettes
    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def getBarPlot(df,pal,barPlotText):
     
    #pal="Viridis" 
    models=list(df.index)
    metrics=list(df.columns)
    
    #color
    colIndex=0
    groupColorPal=all_palettes[pal]
    
    #if no of model is larger than the second largest available color list of corr pallete, use last list that is longest one.
    if len(groupColorPal)<len(models):
        groupColor=groupColorPal[list(groupColorPal.keys())[-1]]
    else:
        #else choose the one with no of color equal to no of models
        if len(models)<3:
            groupColor=groupColorPal[3]
        else:
            groupColor=groupColorPal[len(models)]
        
        
        

    
    #find out number of rows, cols for traces 
    totalrows=round(len(models)/2+0.1)
    totalcols=2
    
    
    fig = make_subplots(rows=totalrows, cols=totalcols,shared_yaxes=True)

    row=1     
    col=1
    #set plot height
    if totalrows<1:
        height=400
    else:
        height=totalrows*250

    
        
    for model in models:
        
        if colIndex>len(groupColor)-3:
            colIndex=0
            
        if len(groupColor)==256:
            colIndex+=round(len(groupColor)/len(models))-1

        #add trace
        if barPlotText:
            trace=go.Bar(name=model, y=metrics, x=list(df.loc[model]),
                         orientation='h', text=list(df.loc[model]),
                             cliponaxis= False,
                             marker=dict(color=groupColor[colIndex])
                             )
        else:
            trace=go.Bar(name=model, y=metrics, x=list(df.loc[model]),
             orientation='h',
                 marker=dict(color=groupColor[colIndex])
                 )
            
        #fig.append_trace(trace, row,col)
        fig.add_trace(trace, row,col)

        #inc rows and col indexer for traces
        if col ==totalcols:
            col=1
            row+=1
        else:
            col+=1
            
        #update color indexer
        colIndex+=1
    
    fig.update_traces(textposition='outside', textfont_size=14)
    fig.update_layout(template="plotly_white",height=totalrows*250,width=1000)
    #fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.05,xanchor="center",x=0.5))
    return dcc.Graph(figure=fig,config=config)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
def getLinePlot(df,pal):
    nmodels=df.shape[0]
    
    if df.shape[1]<3:
        width=400
    else:
        width=df.shape[1]*100
    

    # =============================================================================
    #     #create a list of colors for each line
    # =============================================================================
    if len(all_palettes[pal])<nmodels:
        lineColor=all_palettes[pal][list(all_palettes[pal].keys())[-1]]
        
        #too much color need to remove some of them
        if len(lineColor)==256:
            
            #subset of colors based on totoal no of models
            colSubs=list(range(0,len(lineColor),round(len(lineColor)/nmodels)-1))        
            lineColor =[list(lineColor)[i] for i in colSubs]
        
        #evel longest list has less color than actually needed
        elif len(lineColor)<nmodels:
            lineColor=list(lineColor)*50
      
    else:
        if nmodels<3:
            lineColor=all_palettes[pal][3]
        else:
            lineColor=all_palettes[pal][nmodels]
            
        
        
    
    
    modelName=pd.DataFrame({"Algorithm-Evaluation Method":df.index},index=df.index)
    df = pd.concat([modelName, df], axis=1)
    
    df=pd.melt(df, id_vars =['Algorithm-Evaluation Method'], value_vars =list(df.columns)[1:],
                  var_name ='Metrics', value_name ='Score')
                
    
    
    
    fig = px.line(df, x='Metrics', y='Score', color='Algorithm-Evaluation Method',
                  color_discrete_sequence = lineColor)
    fig.update_layout(template="plotly_white",height=700,width=1050)

# =============================================================================
#     fig.update_layout(legend=dict(
#                             orientation="h",
#                             yanchor="bottom",
#                             y=1.02,
#                             xanchor="right",
#                             x=1
#                         ))
# =============================================================================
    
    
    return dcc.Graph(figure=fig,config=config) 
    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#update subset dropdrown option

modelOtionList=[
        {"label": "All", "value": "all"},
        {"label": "Top 10", "value": "top10"},
            {"label": "Top 20", "value": "top20"},
            {"label": "Top-Bottom 5", "value": "top_bot5"}
            ] 
metricOtionList=options=[{"label": "All", "value": "all"}] 

dropdown={"models":modelOtionList,
          "metric":metricOtionList}
def getOptionList(list_,whichOne):

    dictList=dropdown[whichOne]
 
    temp=[{'label': i, 'value': i} for i in list_]
    return dictList+temp


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#susbet result df based on dropdrown option
def subsetResltDF(df,rows,cols,sortBy):
    nrow=df.shape[0]
    
    #handle first entry
    if cols is None:
        cols=[]
    if rows is None:
        rows=[]
        
    #sorting    
    if (sortBy is None) or (sortBy not in list(df.columns)):
        df=df
    else:
        df=df.sort_values(by=sortBy, ascending=False)
    
    #metric subset
    if ("all" not in cols) and (len(cols)>0):
       results_subset = df[cols]
    else: 
       results_subset = df

    #model subset
    if ("all" in rows) or (len(rows)==0):
        results_subset = results_subset
         
    elif "top10" in rows:
        if nrow>=10:
            results_subset=results_subset.iloc[:10] 
        else:
            results_subset = results_subset
        
    elif "top20" in rows:
        if nrow>=20:
            results_subset=results_subset.iloc[:20]
        else:  
            results_subset = results_subset     
    elif "top_bot5" in rows:
        if nrow>=10:
            select=list(range(0,5,1))+list(range(-5,0,1)) 
            results_subset=results_subset.iloc[select]
        else:  
            results_subset = results_subset 
            
    else:  
        results_subset=results_subset.loc[rows]

    return results_subset

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
#for negative range metrics such as MCC
def bring_To_Positive_Scale(score_list):
    from_min = -1
    from_max = 1
    to_max = 1
    to_min = 0
    score_list_new=[]
    
    for item in score_list:
        score_list_new.append((item - from_min) * (to_max - to_min) / (from_max - from_min) + to_min)
   
    return score_list_new




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#change pkl into dfs
import pickle,sklearn
import pandas as pd
import numpy as np
from numpy import mean

#replace _ with -
def changeColIndex(df): 
                #replace _ with -
    df.columns = df.columns.str.replace(r"_", "-")
    df.index = df.index.str.replace(r"_", "-")
    return df



def getResultDF(trainedModels,refitBy):       
    results_df = pd.DataFrame(index=list(trainedModels.keys())) 
    failedModels={}
    
    for modelName in trainedModels.keys():      
        CV=trainedModels[modelName]                
        
        #check if this model failed
        if isinstance(CV, UserWarning) or isinstance(CV, sklearn.exceptions.FitFailedWarning):
            failedModels[modelName]=str(CV)
            continue
        
        try:
            #check if it is nested CV
            if isinstance(CV, dict):
                results=CV["nested_results"]
                for met in results.keys():
                    if "test_" in met:
                            metName=met.replace("test_","")                                             
                            results_df.loc[modelName,metName]=np.around(mean(results[met]),4)            
            else:
                results=CV.cv_results_     
                rank=list(results["rank_test_"+refitBy]).index(1)     
                for met in results.keys():
                        if "mean_test_" in met:
                            metName=met.replace("mean_test_","")                                             
                            results_df.loc[modelName,metName]=np.around(results[met][rank]  ,4)
        except:
            failedModels[modelName]=str(CV)
            continue
       

            
        
        
    results_NA= results_df[results_df.isna().any(axis=1)]
    results_df.dropna(inplace=True)
    
    failedModels_df=pd.DataFrame(failedModels,index=["Error/Warning"]).T
    
    results_df= results_df.sort_values(refitBy)

    #if there is negative brier score metric or mcc metric, bring it to positive scale
    if "neg_brier_score" in list(results_df.columns):
        results_df['neg_brier_score']=bring_To_Positive_Scale(list(results_df['neg_brier_score']))
      
    if "matthews_corrcoef" in list(results_df.columns):
        results_df['matthews_corrcoef']=bring_To_Positive_Scale(list(results_df['matthews_corrcoef']))
     
    results_df=results_df.sort_values(by=refitBy, ascending=False)    
    results_df=changeColIndex(results_df)
    results_NA=changeColIndex(results_NA)
    failedModels_df=changeColIndex(failedModels_df)
    

    return(results_df,results_NA,failedModels_df)


#!!!!!!!!!!!!!!! Area Plot
def getAreaPlot(df_all,goi,pal):
    #set color list
    groupColorPal=all_palettes[pal]
        
    #if no of goi is larger than the second largest available color list of corr pallete, use last list that is longest one.
    if len(groupColorPal)<len(goi):
        groupColor=groupColorPal[list(groupColorPal.keys())[-1]]
    else:
        #else choose the one with no of color equal to no of goi
        if len(goi)<3:
            groupColor=groupColorPal[3]
        else:
            groupColor=groupColorPal[len(goi)]
                
            
            
    #find out no of classes
    targetClasses=set(df_all.iloc[:,-1])
    targetClasses_tilte = ["<b>Class: "+x +"</b>" for x in targetClasses]

    #make subplots acc
    fig = make_subplots(rows=len(targetClasses), cols=1,vertical_spacing=0.13,subplot_titles=list(targetClasses_tilte))
    
    i=1
    for c in targetClasses:
        
        #subset that particular class
        df = df_all.copy()
        df=df.loc[df_all.iloc[:,-1]== c]
        
        #transform pandas into long
        df["Samples"]=list(df.index)
        df=pd.melt(df, id_vars="Samples", value_vars=goi,var_name='Features', value_name='Value')
        
        temp=px.area(
            df,
            x="Samples",
            y="Value",
            color="Features",
            color_discrete_sequence=groupColor
            
        )
        
        
    
        for trace in range(len(temp["data"])):
            if i!=1:
                temp['data'][trace]['showlegend']=False
            fig.add_trace(temp["data"][trace],row=i, col=1)
        i+=1
    fig.update_layout(template="simple_white",height=350*len(targetClasses),width=1020)    
    return dcc.Graph(figure=fig,config=config)

#!!!!!!!!!!!!!!! BOX Plot
def getBoxPlot(df_all,goi,pal):
    if len(goi)<11:
        width=1020
    else:
        width=100*len(goi)
    
    targetClassName=list(df_all.columns)[-1]
    targetClasses=set(df_all.iloc[:,-1])

    #set color list
    groupColorPal=all_palettes[pal]
        
    #if no of targetClasses is larger than the second largest available color list of corr pallete, use last list that is longest one.
    if len(groupColorPal)<len(targetClasses):
        groupColor=groupColorPal[list(groupColorPal.keys())[-1]]
    else:
        #else choose the one with no of color equal to no of targetClasses
        if len(targetClasses)<3:
            groupColor=groupColorPal[3]
        else:
            groupColor=groupColorPal[len(targetClasses)]
   
   
   #transform pandas into long
    df = df_all.copy()
    df["Samples"]=list(df.index)
    #
    df=pd.melt(df, id_vars=["Samples",targetClassName], value_vars=goi,var_name='Features', value_name='Value')
    
    fig = px.box(df, x="Features", y="Value",
                 #points="all",
                 color_discrete_sequence=groupColor,
                 color=targetClassName, 
                 notched=True,
                 category_orders={"Name": "Value"})

    fig=fig.update_layout(template="simple_white",width=width,height=750)
    
    return dcc.Graph(figure=fig,config=config)

#!!!!!!!!!!!!!
#Distribution Plot
import plotly.figure_factory as ff
def getDistPlot(df_all,goi,pal,curve_type):
    
    groupColorPal=all_palettes[pal]
            
    #if no of goi is larger than the second largest available color list of corr pallete, use last list that is longest one.
    if len(groupColorPal)<len(goi):
        groupColor=groupColorPal[list(groupColorPal.keys())[-1]]
    else:
        #else choose the one with no of color equal to no of goi
        if len(goi)<3:
            groupColor=groupColorPal[3]
        else:
            groupColor=groupColorPal[len(goi)]
                    
                
    #find out no of classes
    targetClasses=set(df_all.iloc[:,-1])
    targetClasses_tilte = ["<b>Class: "+x +"</b>" for x in targetClasses]

    #make subplots acc
    fig = make_subplots(rows=len(targetClasses), cols=1,vertical_spacing=0.13,subplot_titles=list(targetClasses_tilte))
    
    i=1
                
    for c in targetClasses:
            
            #subset that particular class
            df = df_all.copy()
            df=df.loc[df_all.iloc[:,-1]== c]
            
            #generate plot
            histData=[]
            for feat in goi:
                histData.append(df[feat].tolist())
                
            temp=ff.create_distplot(histData, goi, show_hist=False,
                                    curve_type=curve_type,colors=groupColor)
            
            temp['layout']['xaxis']['title']='Feature Value'
            
            #append traces
            for trace in range(len(temp["data"])):
                if i!=len(targetClasses):
                    temp['data'][trace]['showlegend']=False
                fig.add_trace(temp["data"][trace],row=i, col=1)
            
            fig.update_yaxes(title_text="Density", row=i, col=1)   
            fig.update_xaxes(title_text="Feature Value", row=i, col=1)
            i+=1

                
    fig=fig.update_layout(template="simple_white", 
                            height=750,width=1020)
            
    #fig.show()
        
    return dcc.Graph(figure=fig,config=config)
        
#!!!!!!!!
#scatter matrix
def getScatterMatrix(df_all,goi,pal,diag_type):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if len(goi)<5:
        width=1020
        height=750
    else:
        width=200*len(goi)
        height=200*len(goi)
        
        
    targetClassName=list(df_all.columns)[-1]
    targetClasses=set(df_all.iloc[:,-1])
    
    #set color list
    groupColorPal=all_palettes[pal]
        
    #if no of targetClasses is larger than the second largest available color list of corr pallete, use last list that is longest one.
    if len(groupColorPal)<len(targetClasses):
        groupColor=groupColorPal[list(groupColorPal.keys())[-1]]
    else:
        #else choose the one with no of color equal to no of targetClasses
        if len(targetClasses)<3:
            groupColor=groupColorPal[3]
        else:
            groupColor=groupColorPal[len(targetClasses)]
                
    
    df = pd.concat([df_all[goi], df_all.iloc[:,-1]], axis=1)
    fig = ff.create_scatterplotmatrix(df, diag=diag_type, index=targetClassName,
                                     colormap=groupColor,width=width,height=height)
    #return fig
    return dcc.Graph(figure=fig,config=config)

 

#!!!!!!!!
#Class Distribution
def getClasssDist(df_all,pal):
    #set color list
    groupColorPal=all_palettes[pal][8]
    
    targetClasses_count=pd.DataFrame(df_all.iloc[:,-1].value_counts())
    targetClasses_count.columns = ['No of Samples Per Class']
    
    targetClasses_count["Class"]=list(targetClasses_count.index)
    
    
    
    fig = fig = px.bar(targetClasses_count, x='Class', y='No of Samples Per Class',text_auto=True)
    fig=fig.update_layout(template="simple_white", 
                        height=500,width=1020)
    
    fig.update_traces(marker_color=groupColorPal)
    #fig.show()
    return dcc.Graph(figure=fig,config=config)


from sklearn import set_config
from sklearn.utils import estimator_html_repr
import dash_bootstrap_components as dbc

filename_pipe=""
def getPieline(trainedModels,models,config):

    models=models.replace("-", "_")
    
    #if config=="diagram":
    #del all file from templ loc if there are too much
    filelist = [ f for f in os.listdir("./assets/") if f.endswith(".html") ]
    if len(filelist)>5:
        for f in filelist:
            os.remove(os.path.join("./assets/", f))
            
    #create filenames
    current_time = datetime.now().strftime("%H_%M_%S")
    fileName="./assets/pipeline_"+models+"_"+current_time+".html"
    
    set_config(display='diagram')
    with open(fileName, 'w') as f:  
        f.write(estimator_html_repr(trainedModels[models]))

    a=dbc.Row([                    
                dbc.Col([        
                        html.Div(dbc.Button(html.I(" Download", className="fa fa-solid fa-download"), color="primary",id='pipeDownload', className="me-1", 
                                 style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                                 className="d-grid gap-2 d-md-flex justify-content-md-end",),
                             dcc.Download(id="download-pipe")
                    ],width=2),
                dbc.Col(html.Iframe(id='target',
                        src=fileName,
                        style={"height": "800px", "width": "100%","border":"0px"}),width=10),

                    ])
    global filename_pipe
    filename_pipe=fileName
    return a
# =============================================================================
#     else:
#         set_config(display='text')
# 
#         return html.Div(str(trainedModels[models]))
# =============================================================================

#!!!!!!!!
#Download pipe
from dash.dependencies import Input, Output,State,ALL
from app import app

@app.callback(
    Output("download-pipe", "href"), 
    Input("pipeDownload","n_clicks") 
) 

def down_pipe(n_clicks): 
    if n_clicks:  
        return dcc.send_file(filename_pipe)


"""import pandas as pd
data=pd.read_csv("/Users/akshay/OneDrive - Universitaet Bern/PhD/Projetcs-extra/easyML/testDatasetsResults/TCGA-BRCA_mRNA/TCGA-BRCA_new.csv",index_col=0,sep=",")
X= data.iloc[:,0:-1]
y = data.iloc[:,-1]
X.shape  
from sklearn.feature_selection import SelectKBest,VarianceThreshold

var_thr = VarianceThreshold(threshold = 0.5) #Removing both constant and quasi-constant
var_thr.fit(X)
concol = [column for column in X.columns 
      if column not in X.columns[var_thr.get_support()]]

for features in concol:
    print(features)
X.drop(concol,axis=1,inplace=True)
X.shape"""




# =============================================================================
# 
# 
# import pickle
# import pandas as pd 
# with open('/Users/akshay/Downloads/data_12_56_22/userInputData/inputParameters_12_56_22.pkl', 'rb') as handle:
#     userInputData=pickle.load(handle)
#  
# =============================================================================
"""import pickle
import sklearn
import numpy as np
import math
with open("/Users/akshay/Downloads/userInputData 3/inputParameters_13_57_46.pkl", 'rb') as handle:
    trainedModels=pickle.load(handle)
            

trainedModels.keys()
del trainedModels["modelEval_tab_active"]["NestedCV"]"""

# =============================================================================
# # plotly
# import plotly.io as pio
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.figure_factory as ff
# from bokeh.palettes import all_palettes
# pio.renderers.default = 'browser'
# import pandas as pd    
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
# inputData=pd.read_csv("cll.txt",index_col=0,sep="\t")
# pal="Turbo"
# df_all = inputData.copy()
# 
# =============================================================================
#getAreaPlot(inputData,list(inputData.columns)[1:8],pal)
#getBoxPlot(inputData,list(inputData.columns)[1:10],pal)
#getDistPlot(inputData,list(inputData.columns)[1:10],pal,"kde",10)
#getScatterMatrix(inputData,list(inputData.columns)[1:10],pal,"histogram")
#getClasssDist(inputData,pal)
"""
import pandas as pd
data = pd.read_csv('/Users/akshay/OneDrive - Universitaet Bern/PhD/Projetcs-extra/easyML/testDatasetsResults/cervical/cervical.csv',index_col=0)
X= data.iloc[:,0:-1]
y = data.iloc[:,-1]
    
stats=pd.DataFrame()
stats["mean"]=X.mean()
stats["Std.Dev"]=X.std()
stats["Var"]=X.var()"""

