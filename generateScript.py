#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:19:43 2022

@author: akshay
"""


from UI.componentIDs import classification_Com_IDS,classification_models,\
    undersampling_Com_IDS,underSamp_models,\
       overrsampling_Com_IDS, overSamp_models, \
           modelEval_Com_IDS,\
               scaling_Com_IDS,scaling_models, \
                   featSel_Com_IDS, featSel_models,featSel_est

from helperFunctions import getAlgoNames,removeModelId,getActiveAlgo,getMoedlEvalActive,saveUserInputData,getActiveAlgoFeatSel
import pickle

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# importing the module
import ast
with open('myfile.txt') as f:
    data = f.read()
userInputData=ast.literal_eval(data)  
#a=saveUserInputData(userInputData)

#get random state4  
if("random_seed" in userInputData.keys()):
    rs=userInputData["random_seed"]
else:
    rs=12345

#set numpy random seed
import numpy as np
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

userInputData={"random_state":rs,"n_jobs":userInputData["n_jobs"],\
      "scaling_tab_active":scaling_tab_active,"underSamp_tab_active":underSamp_tab_active,\
      "overSamp_tab_active":overSamp_tab_active,"classification_tab_active":classification_tab_active,
      "featSel_tab_active":userInputData["featSel_tab_para"],\
      "modelEval_tab_active":modelEval_tab_active,\
      "modelEval_metrices":userInputData["modelEval_metrices_tab_para"][0]
}
    
#Save user input data as pkl object
with open('userInputData.pkl', 'wb') as handle:
    pickle.dump(userInputData, handle)

with open('userInputData_1.pkl', 'rb') as handle:
    userInputData=pickle.load(handle)

