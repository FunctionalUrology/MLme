#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:22:00 2022

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
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn.feature_selection import VarianceThreshold


from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn import metrics

def getTestScores(whichMetrics,y_true,yPred,k):

    if k<3:
        metrics_={'accuracy': accuracy_score(y_true, yPred),
     'balanced_accuracy': balanced_accuracy_score(y_true, yPred),
     'average_precision': average_precision_score(y_true, yPred),
     'f1': f1_score(y_true, yPred),
     'f1_micro': f1_score(y_true, yPred,average='micro'),
     'f1_weighted': f1_score(y_true, yPred,average='weighted'),
     'f1_macro': f1_score(y_true, yPred,average='macro'),
     'matthews_corrcoef': matthews_corrcoef(y_true, yPred),
     'jaccard': jaccard_score(y_true, yPred),
     'precision': precision_score(y_true, yPred),
     'recall': recall_score(y_true, yPred),
     'top_k_accuracy': top_k_accuracy_score(y_true, yPred,k=k),
     'roc_auc': roc_auc_score(y_true, yPred)}

    else:
        metrics_={'accuracy': metrics.accuracy_score(y_true, yPred),
     'balanced_accuracy': metrics.balanced_accuracy_score(y_true, yPred),

      'f1_micro': metrics.f1_score(y_true, yPred,average='micro'),
      'f1_weighted': metrics.f1_score(y_true, yPred,average='weighted'),
      'f1_macro': metrics.f1_score(y_true, yPred,average='macro'),
      'matthews_corrcoef': metrics.matthews_corrcoef(y_true, yPred),
      'jaccard': metrics.jaccard_score(y_true, yPred,average="micro"),
      'precision': metrics.precision_score(y_true, yPred,average="micro"),
      'recall': metrics.recall_score(y_true, yPred,average="micro")
     }

    testScores={}

    for metric in whichMetrics:
        if metric in metrics_.keys():
            testScores[metric]=metrics_[metric]

    return testScores

# =============================================================================
# #Setup argument switch
# =============================================================================
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Parse command line arguments
parser = ArgumentParser(description="Execute the custom machine learning pipeline from customML.",
                        formatter_class=ArgumentDefaultsHelpFormatter)
requiredNamed = parser.add_argument_group('required arguments')

requiredNamed.add_argument("-i", "--input",
                    help="A .csv or .txt file with a row representing a sample and a column representing a feature. The first and last columns must contain the sample name and target classes, respectively, and the file must not have any NaN values. ")

requiredNamed.add_argument("-s", "--separator",
                    help="Delimiter to use for input file. Possible options are 'comma' and 'tab'. Please use the keywords not the symbols. For example use 'comma' not ','.")

requiredNamed.add_argument("-p", "--parameters",
                    help="A pickle file (inputParameter.pkl) from customML.")

parser.add_argument("-o", "--output", default=".",
                    help="A directory where all the results will be saved.")
args = vars(parser.parse_args())

# =============================================================================
# #check input data exist
# =============================================================================
inputFile=args["input"]
separatorInput=args["separator"]
inputParameterFile=args["parameters"]
outputPath=args["output"]


##check inputFile
if inputFile==None:
    print("\n Following error occured: \n\t\t"+" Please provide input file using -i tag.")
    sys.exit(1)

if os.path.exists(inputFile)==False:
    print("\n Following error occured: \n\t\t"+inputFile + " file does not exist.")
    sys.exit(1)

# #check separatorInput
if separatorInput==None:
    print("\n Following error occured: \n\t\t"+" Please provide file delimiter using -s tag.")
    sys.exit(1)

if separatorInput!="comma" and separatorInput!="tab":
    print("\n Following error occured: \n\t\t The given delimiter is not valid. Possible options are 'comma' and 'tab'. Please use the keywords not the symbols. For example use 'comma' not ','")
    sys.exit(1)

# #check inputParameterFile
if inputParameterFile==None:
    print("\n Following error occured: \n\t\t"+" Please provide parameter file (inputParameter.pkl) using -p tag.")
    sys.exit(1)

if os.path.exists(inputParameterFile)==False:
    print("\n Following error occured: \n\t\t"+inputParameterFile + " file does not exist.")
    sys.exit(1)

if inputParameterFile.endswith(".pkl")==False:
    print("\n Following error occured: \n\t\t"+inputParameterFile + " is not a pickle (pkl) file. Please provide a pickle file (inputParameter.pkl) from customML.")
    sys.exit(1)

# #check outputPath
if os.path.exists(outputPath)==False:
    print("\n Following error occured: \n\t\t"+outputPath + " directory does not exist.")
    sys.exit(1)

if outputPath!="." and os.path.isdir(outputPath)==False:
    print("\n Following error occured: \n\t\t"+outputPath+" is not a directory. Please provide a directory.")
    sys.exit(1)

#if it is . the getcwd
if outputPath==".":
    outputPath=os.getcwd()


#set seprators symbol
if separatorInput=="comma":
    separatorInput=","
else:
    separatorInput="\t"



# =============================================================================
# #write logs
# =============================================================================
date = datetime.now().strftime("%I_%M_%S_%p-%d_%m_%Y")

#logFolder = os.path.join(os.getcwd(),"logs")
logFolder = os.path.join(outputPath,'result') #date+

filename=os.path.join(logFolder,'log.txt')

if os.path.exists(logFolder):
    sys.stdout = open(filename, 'w')
else:
    os.makedirs(logFolder)
    sys.stdout = open(filename, 'w')


# =============================================================================
# #read data
# =============================================================================

print("=============================================================================")
print("\t\tInput data/arguments")
print("=============================================================================")
print(args)

print("=============================================================================")
print("\t\tReading data start...")
print("=============================================================================")

try:

    data=pd.read_csv(inputFile,sep=separatorInput,index_col=0)
except Exception as e:
    print(e)
    sys.exit(1)

X= data.iloc[:,0:-1]
y = data.iloc[:,-1]

print("-----------------------------------------")
print("Data info \n",
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


# =============================================================================
# read user input data
# =============================================================================
print("\n\n\n\n=============================================================================")
print("\t\tReading user input data...")
print("=============================================================================\n\n\n\n")



# =============================================================================
# with open(inputParameterFile, 'rb') as handle:
#     userInputData=pickle.load(handle)
# =============================================================================

try:

    with open(inputParameterFile, 'rb') as handle:
        userInputData=pickle.load(handle)
except Exception as e:
    print(e)
    sys.exit(1)

print(userInputData)

random_state=userInputData["random_state"]
n_jobs=userInputData["n_jobs"]
scaling_tab_active=userInputData["scaling_tab_active"]
overSamp_tab_active=userInputData["overSamp_tab_active"]
underSamp_tab_active=userInputData["underSamp_tab_active"]
classification_tab_active=userInputData["classification_tab_active"]
featSel_tab_active=userInputData["featSel_tab_active"]
modelEval_tab_active=userInputData["modelEval_tab_active"]
modelEval_metrices=userInputData["modelEval_metrices"]
refit_Metric=userInputData["refit_Metric"][0]
indepTestSet=userInputData["indepTestSet"]

# Train-test split, intentionally use shuffle=False
if indepTestSet!=None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=indepTestSet, random_state=random_state,shuffle=True)
    X,y=X_train,y_train

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

#change k for top k acc
if "top_k_accuracy" in list(modelEval_metrices.keys()):
    modelEval_metrices["top_k_accuracy"]=make_scorer(top_k_accuracy_score,k=len(counter)-1)

#check if it is a multiclass. If yes change the average parameter
if len(counter)>2:
    modelEval_metrices.pop("roc_auc", None)
    modelEval_metrices.pop("f1", None)
    modelEval_metrices.pop("average_precision", None)
    modelEval_metrices.pop("top_k_accuracy", None)
    modelEval_metrices.pop("neg_brier_score", None)


    if "precision" in list(modelEval_metrices.keys()):
        modelEval_metrices["precision"]=make_scorer(precision_score,average="micro")
    if "recall" in list(modelEval_metrices.keys()):
        modelEval_metrices["recall"]=make_scorer(recall_score,average="macro")
    if "jaccard" in list(modelEval_metrices.keys()):
        modelEval_metrices["jaccard"]=make_scorer(jaccard_score,average="macro")
    if refit_Metric not in list(modelEval_metrices.keys()):
        refit_Metric=list(modelEval_metrices.keys())[0]


print("\n\n\n\nUser Input Data: Successfull")
print("=============================================================================")
print("=============================================================================\n\n\n\n")

print("=============================================================================")
print("\t\tPipeline initialization and Model evalaution start.")
print("=============================================================================\n\n\n\n")

#set random seed for numpy
np.random.seed(random_state)


# =============================================================================
# create pipeline and evaluate each model
# =============================================================================
trainedModels={}
featureIndex_name={}
testScore={}


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
        pipe_list=[('vt', VarianceThreshold())]

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

# =============================================================================
#         parameters = {'scaler': scalers,       #£££££££££££ Remove in final version
#                   'sampler': samplers}         #£££££££££££ Remove in final version
#         pipe = Pipeline_imb([
#                     ('scaler',  scalers[0]),
#                     ('sampler', samplers[0]),
#                     ('classifier', model)
#                 ])
# =============================================================================

    #get CV method
    for modelEval in modelEval_tab_active:

        cv=modelEval_tab_active[modelEval]

        #handle unexpected error
        try:
            print("----------------------------")
            print("MODEL:  "+modelName.upper()+" and "+modelEval.upper())

            grid = GridSearchCV(pipe, parameters, cv=cv,n_jobs=n_jobs,
                                refit=refit_Metric,scoring = modelEval_metrices,return_train_score=True)

            if modelEval=="NestedCV":
                nested_scores = cross_validate(grid, X, y, scoring=modelEval_metrices,cv=cv, n_jobs=n_jobs)
                grid=grid.fit(X, y)

                trainedModels[modelName+"_"+modelEval]={}
                trainedModels[modelName+"_"+modelEval]["grid"]=grid
                trainedModels[modelName+"_"+modelEval]["nested_results"]=nested_scores

                #test score
                if indepTestSet!=None:
                    y_pred = grid.predict(X_test)
                    testScore[modelName+"_"+modelEval]=getTestScores(modelEval_metrices.keys(),y_test,y_pred,len(counter)-1)



                print("Best Score for given refit metric ("+refit_Metric+") :  "+str(grid.best_score_))
                print("\n\n\t\t")

                if "featSel" in grid.best_estimator_.named_steps.keys():
                    featureIndex_name[modelName+"_"+modelEval]=X.iloc[:,grid.best_estimator_.named_steps['featSel'].get_support(indices=True)].columns.tolist()


            else:
                grid=grid.fit(X, y)
                trainedModels[modelName+"_"+modelEval]=grid

                #test score
                #you might need to comment it out.
                if indepTestSet!=None:
                    y_pred = grid.predict(X_test)
                    testScore[modelName+"_"+modelEval]=getTestScores(modelEval_metrices.keys(),y_test,y_pred,len(counter)-1)

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



trainedModels["featSel_name"]=featureIndex_name
trainedModels["testScore"]=testScore

print("=============================================================================")
print("\t\tPipeline Done")
print("=============================================================================\n\n\n\n")

#Save user input data as pkl object
#fileName=inputFile.split(".")[0]+"_"+'trainedModels.pkl'
fileName=os.path.join(logFolder,'results.pkl')
with open(fileName, 'wb') as handle:
    pickle.dump(trainedModels, handle)

sys.stdout.close()

#zip result folder
import os, zipfile,shutil

name = logFolder
zip_name = name + '.zip'

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
    for folder_name, subfolders, filenames in os.walk(name):
        for filename in filenames:
            file_path = os.path.join(folder_name, filename)
            zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))

#remove result folder after zipping
shutil.rmtree(logFolder)
