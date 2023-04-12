#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:27:00 2022

@author: akshay
"""

# =============================================================================
# Import packages
# =============================================================================
import pandas as pd
import os
from collections import Counter
import pickle
import numpy as np


import warnings
warnings.filterwarnings("error")

from datetime import datetime
import sys

from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,Normalizer,\
    PowerTransformer,QuantileTransformer,RobustScaler,StandardScaler,LabelEncoder

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier,GradientBoostingClassifier

from imblearn.pipeline import Pipeline as Pipeline_imb

from sklearn.model_selection import *
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from sklearn.feature_selection import SelectPercentile,VarianceThreshold

def runSubscript(data,date,varTH_automl,percentile):
    #!!!!!!!!!! Input Data
    random_state=123
    #set random seed for numpy
    np.random.seed(random_state)  
    n_jobs=-1
    
    scaling_tab_active={'MaxAbs Scaler': MaxAbsScaler(),'MinMax Scaler': MinMaxScaler()}
    overSamp_tab_active={'RandomOverSampler': RandomOverSampler(random_state=123)}
    underSamp_tab_active={'RandomUnderSampler': RandomUnderSampler(random_state=123)}
    
    classification_tab_active={'Dummy Classifier': DummyClassifier(random_state=123),
     'SVM': SVC(probability=True, random_state=123),
     'KNN': KNeighborsClassifier(n_jobs=-1, p=1),
     'AdaBoost': AdaBoostClassifier(random_state=123),
     'GaussianNB': GaussianNB()}
    
    featSel_tab_active={'SelectPercentile': SelectPercentile(percentile=percentile)}
    
    modelEval_tab_active={'RepeatedStratifiedKFold': RepeatedStratifiedKFold(n_repeats=10, n_splits=5, random_state=None),
     'StratifiedShuffleSplit': StratifiedShuffleSplit(n_splits=10, random_state=None, test_size=None,
                 train_size=None),
     'NestedCV': StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
     }
    
    modelEval_metrices=['accuracy','average_precision','f1','balanced_accuracy','f1_macro','f1_micro',
                        'f1_weighted','jaccard','precision','matthews_corrcoef','recall','roc_auc','top_k_accuracy']
    refit_Metric='average_precision'
    
    
    #!!!!!!!!!! Handle metrics for multiclass
    #check if mcc is there, if yes make a scoring fucntion
    if "matthews_corrcoef" in modelEval_metrices:
        modelEval_metrices = dict(zip(modelEval_metrices, modelEval_metrices))
        modelEval_metrices["matthews_corrcoef"]=make_scorer(matthews_corrcoef)
    else:
        modelEval_metrices = dict(zip(modelEval_metrices, modelEval_metrices))
        
    #change average for f1
    if "f1_micro" in list(modelEval_metrices.keys()):
        modelEval_metrices["f1_micro"]=make_scorer(f1_score,average="micro")
    if "f1_macro" in list(modelEval_metrices.keys()):
        modelEval_metrices["f1_macro"]=make_scorer(f1_score,average="macro")
    if "f1_weighted" in list(modelEval_metrices.keys()):
        modelEval_metrices["f1_weighted"]=make_scorer(f1_score,average="weighted")
        



    # =============================================================================
    # #write logs
    # =============================================================================
             
    #date = datetime.now().strftime("%I_%M_%S_%p-%d_%m_%Y")    
    os.makedirs(os.path.join(os.getcwd(),"autoML_output/"+date))
    logFolder = os.path.join(os.getcwd(),"autoML_output/"+date)
    filename=os.path.join(logFolder,date+'-log.txt')
    
    if os.path.exists(logFolder):
        sys.stdout = open(filename, 'w')
    else:
      os.makedirs(logFolder)
      sys.stdout = open(filename, 'w')
      
    
    # =============================================================================
    # #read data
    # =============================================================================
    print("=============================================================================")
    print("\t\tReading data start...")
    print("=============================================================================")


    X= data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    
    print("-----------------------------------------")
    print("Data info before variance based filtering: \n",
         "\t\tData: ",X.shape,
          "\n\t\tNumber of NA values in data: ",X.isna().sum().sum(),
         "\n\t\tTarget Variable", y.shape,
         "\n\t\tNumber of NA values in Target Variable: ",X.isna().sum().sum())
    
    print("-----------------------------------------")
    print("Following constant features will be removed:")
    var_thr = VarianceThreshold(threshold = varTH_automl) #Removing both constant and quasi-constant
    var_thr.fit(X)
    concol = [column for column in X.columns 
          if column not in X.columns[var_thr.get_support()]]

    for features in concol:
        print(features)
    X.drop(concol,axis=1,inplace=True)
    

    print("-----------------------------------------")
    print("Data info after variance based filtering: \n",
          "\t\tData: ",X.shape,
           "\n\t\tNumber of NA values in data: ",X.isna().sum().sum(),
          "\n\t\tTarget Variable", y.shape,
          "\n\t\tNumber of NA values in Target Variable: ",X.isna().sum().sum())
    
    print("Reading Data complete.")
    print("=============================================================================")
    print("=============================================================================\n\n\n\n")
    
    print("=============================================================================")
    print("\t\tClass Distribution.")
    print("=============================================================================")
    
    #encode target variable
    y = LabelEncoder().fit_transform(y)
    
    # summarize distribution
    counter = Counter(y)
    for k,v in counter.items():
     	per = v / len(y) * 100
     	print('\t\tClass=%d, n=%d (%.3f%%)' % (k, v, per))
    
    print("\n\n\n\n=============================================================================")
    
    #change k for top k acc
    if "top_k_accuracy" in list(modelEval_metrices.keys()):
        modelEval_metrices["top_k_accuracy"]=make_scorer(top_k_accuracy_score,k=len(counter)-1)
        
    #check if it is a multiclass. If yes change the average parameter
    if len(counter)>2:
        modelEval_metrices.pop("roc_auc", None)
        modelEval_metrices.pop("f1", None)
        modelEval_metrices.pop("average_precision", None)
        modelEval_metrices.pop("top_k_accuracy", None)
    
        if "precision" in list(modelEval_metrices.keys()):
            modelEval_metrices["precision"]=make_scorer(precision_score,average="macro")
        if "recall" in list(modelEval_metrices.keys()):
            modelEval_metrices["recall"]=make_scorer(recall_score,average="macro")
        if "jaccard" in list(modelEval_metrices.keys()):
            modelEval_metrices["jaccard"]=make_scorer(jaccard_score,average="macro")
        if refit_Metric not in list(modelEval_metrices.keys()):
            refit_Metric=list(modelEval_metrices.keys())[0]
        
    # =============================================================================
    # create pipeline and evaluate each model
    # =============================================================================
    print("=============================================================================")
    print("\t\tPipeline initialization and Model evalaution start.")
    print("=============================================================================\n\n\n\n")
    
    trainedModels={}
    featureIndex_name={}
    
    trainedModels["refit_Metric"]=refit_Metric
    
    warnings.filterwarnings("error")
    print("\t\t\t\t**** Model name and Evaluation method ****\n\n")
    
    for modelName in list(classification_tab_active.keys()):
        model=classification_tab_active[modelName]
    
        #get scaling and sampling info
        scalers=list(scaling_tab_active.values())
        samplers=list(overSamp_tab_active.values())+list(underSamp_tab_active.values())
        featSel=list(featSel_tab_active.values())
    
    
        #if it is dummy, do not perform any preprocessing
        if(modelName=="Dummy Classifier"):
            parameters = {'classifier': [model]}
            pipe = Pipeline_imb([('classifier', model)])
    
        else:
    
            parameters={}
            pipe_list=[('vt', VarianceThreshold(varTH_automl))]
            
            if len(scalers)>0 and scalers[0]!=[]:
                parameters['scaler'] = scalers
                pipe_list.append(('scaler',  scalers[0]))
    
            if len(samplers)>0 and samplers[0]!=[]:
                parameters['sampler'] = samplers
                pipe_list.append(('sampler', samplers[0]))
    
            if len(featSel)>0 and featSel[0]!=[]:
                    parameters['featSel'] = featSel
                    pipe_list.append(('featSel', featSel[0]))
    
            pipe_list.append(('classifier', model))
            pipe = Pipeline_imb(pipe_list)
        
        #get CV method
        for modelEval in modelEval_tab_active:
    
            cv=modelEval_tab_active[modelEval]
    
            #handle unexpected error
            try:
                print("----------------------------")
                print("MODEL:  "+modelName.upper()+" and "+modelEval.upper())
                
                
                grid = GridSearchCV(pipe, parameters, cv=cv,#n_jobs=n_jobs,
                                    refit=refit_Metric,scoring = modelEval_metrices,return_train_score=True)
    
                if modelEval=="NestedCV":
                    nested_scores = cross_validate(grid, X, y, scoring=modelEval_metrices,cv=cv, n_jobs=n_jobs)
                    #with np.errstate(divide='ignore'):
                    grid=grid.fit(X, y)
    
                    trainedModels[modelName+"_"+modelEval]={}
                    trainedModels[modelName+"_"+modelEval]["grid"]=grid
                    trainedModels[modelName+"_"+modelEval]["nested_results"]=nested_scores
                    print("Best Score for given refit metric:  "+str(grid.best_score_))
                    print("\n\n\t\t")
                    
                    if "featSel" in grid.best_estimator_.named_steps.keys():
                        featureIndex_name[modelName+"_"+modelEval]=X.iloc[:,grid.best_estimator_.named_steps['featSel'].get_support(indices=True)].columns.tolist()
    
    
                else:
                    grid=grid.fit(X, y) 
                    trainedModels[modelName+"_"+modelEval]=grid
                    print("Best Score for given refit metric ("+refit_Metric+") :  "+str(grid.best_score_))
                    print("\n\n\t\t")
                    
                    if "featSel" in grid.best_estimator_.named_steps.keys():
                        featureIndex_name[modelName+"_"+modelEval]=X.iloc[:,grid.best_estimator_.named_steps['featSel'].get_support(indices=True)].columns.tolist()
                
            
            except Exception as e:
                trainedModels[modelName+"_"+modelEval]=e
                print("\n\t\t!!!!!!!!!!!!!!!!")
                print(modelName+" or "+modelEval+" failed due to following error: \n")
                print(e)
                print("\n\nPIPELINE:\n")
                print(grid)
                print("\n\n")
            
            break
            

    
    
    
    trainedModels["featSel_name"]=featureIndex_name
    
    print("=============================================================================")
    print("\t\tPipeline Done")
    print("=============================================================================\n\n\n\n")
    
    #Save user input data as pkl object
    fileName=logFolder+'/trainedModels.pkl'
    with open(fileName, 'wb') as handle:
        pickle.dump(trainedModels, handle)
    
    #ßreturn date
         
         