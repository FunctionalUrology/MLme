#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:20:12 2021

@author: akshay
"""


import dash_bootstrap_components as dbc
from dash import dcc,html
from UI.scaling import genrateInfoCallback, genrateCollapseCallback,genrateAlertCallback,infoText
from UI.dataSampling import get_neighbors_Para
from app import app
from dash.dependencies import Input, Output,State

    #get the header of each card of sampling tab
def getAlgoHeader(algo):
    return dbc.Row([
     dbc.Col(dbc.Checklist(options=[{"label": algo,"value": True}],
                                    value=[],
                                    id=algo,
                                    inline=True,switch=True,labelStyle={"font-weight": "bold",
                                                                        "font-size": "18px"},
                                    labelCheckedStyle={"color": "green"},persistence=True,persistence_type="memory"),
              width={"size": 9},),
   dbc.Col(dbc.Button("Parameters",id=algo+"-collapse-button",
            className="mr-1",size="sm",color="light",n_clicks=0,style={"margin-top": "15px"}),
        width={"size": 3},)
       ])

  #get the Integer input of sampling tab
def getInputNumberDiv(title,text,id,placeholder,step,minimum):
    component= dbc.FormGroup([
                
                    html.Div(dbc.Label(title,style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    html.Div(dbc.Label(text,style={"margin-top": "12px","font-size": "10px",})),
                    dbc.Input(type="number",min=minimum,placeholder=placeholder,id=id,persistence=True,persistence_type="memory"),
                    ])

    return component



#define card for classification tab
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

dummy_content=[
    dbc.CardHeader(getAlgoHeader("Dummy Classifier")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            dbc.Label("Strategy to use to generate predictions",style={"font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "Stratified:  generates predictions by respecting the training set’s class distribution.", "value": "stratified"},
                                    {"label": "Most frequent: always predicts the most frequent label in the training set.", "value": "most_frequent"},
                                    {"label": "Prior: always predicts the class that maximizes the class prior.", "value": "prior"},
                                    {"label": "Uniform: generates predictions uniformly at random.", "value": "uniform"},
                                ],value='prior', clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="Dummy Classifier-strategy",persistence=True,persistence_type="memory")
                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="Dummy Classifier-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("Dummy Classifier"),
                      id="Dummy Classifier-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="Dummy Classifier-collapse",is_open=False,),
    ]


SVM_content=[
    dbc.CardHeader(getAlgoHeader("SVM")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            html.Div(dbc.Label("C: Regularization parameter",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty. ",style={"margin-top": "12px","font-size": "10px",})),
            dbc.Input(type="number",placeholder=1, min=0,id="SVM-C",persistence=True,persistence_type="memory"),

            dbc.FormGroup([
                            dbc.Label("Kernel",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "linear", "value": "linear"},{"label": "poly", "value": "poly"},
                                    {"label": "rbf", "value": "rbf"},{"label": "sigmoid", "value": "sigmoid"},
                                    
                                ],value='rbf', clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SVM-kernel",persistence=True,persistence_type="memory")
                         ]),
            
            getInputNumberDiv("Degree",
                              "Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.",
                              "SVM-degree",3,1,1),
 
            html.Div(dbc.Label("Gamma",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
            html.Div(dbc.Label("Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. ",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(options=[
                                    {"label": "Scale", "value": "scale"},
                                    {"label": "Auto", "value": "auto"}
                                    ],value="scale",id="SVM-gamma"
                                    ,persistence=True,persistence_type="memory"),
            
                                 
            dbc.FormGroup([
                            dbc.Label("Tolerance for stopping criterion",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("Absolute threshold for a singular value of X to be considered significant, used to estimate the rank of X. Dimensions whose singular values are non-significant are discarded. Only used if solver is ‘svd’.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.001", "value": 0.001},
                                    {"label": "0.0001", "value": 0.0001},
                                    {"label": "0.00001", "value": 0.00001},
                                ],value=0.001, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SVM-tol",persistence=True,persistence_type="memory")

                         ]),
            
                                        
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SVM-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SVM"),
                      id="SVM-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SVM-collapse",is_open=False,),
    ]

KNN_content=[
    dbc.CardHeader(getAlgoHeader("KNN")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            get_neighbors_Para("N Neighbors",
                   "Number of neighbors to use by default for kneighbors queries.",
                   "KNN-n_neighbors",5),
            
            dbc.Label("Weights: weight function used in prediction",style={"font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "Uniform: all points in each neighborhood are weighted equally.", "value": "uniform"},
                                    {"label": "Distance: weight points by the inverse of their distance. In this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.", "value": "distance"},
                                ],value='uniform', clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="KNN-weights",persistence=True,persistence_type="memory"),
                            
   
            dbc.Label("Algorithm",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Algorithm used to compute the nearest neighbors.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "Ball Tree", "value": "ball_tree"},
                    {"label": "Kd Tree", "value": "kd_tree"},
                    {"label": "Brute", "value": "brute"},
                    {"label": "Auto", "value": "auto"},
                    
                ],value='auto',id="KNN-algorithm",
                persistence=True,persistence_type="memory"),
                     
            getInputNumberDiv("Leaf Size",
                              "Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
                              "KNN-leaf_size",2,1,1),
            
            dbc.Label("P: Power parameter for the Minkowski metric.",style={"font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "1: equivalent to using manhattan_distance.", "value": 1},
                                    {"label": "2: equivalent to using euclidean_distance. ", "value": 2},
                                ],value=1, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="KNN-p",persistence=True,persistence_type="memory"),
                            
                        
                                        
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="KNN-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("KNN"),
                      id="KNN-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="KNN-collapse",is_open=False,),
    ]

ExtraTree_content=[
    dbc.CardHeader(getAlgoHeader("ExtraTree")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
            
            dbc.Label("Criterion",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("The function to measure the quality of a split.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "Gini", "value": "gini"},
                    {"label": "Entropy", "value": "entropy"},                    
                ],value='gini',id="ExtraTree-criterion",
                persistence=True,persistence_type="memory"),
            

                        
            dbc.Label("Splitter",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("The strategy used to choose the split at each node.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "Random", "value": "random"},
                    {"label": "Best", "value": "best"},                    
                ],value='random',id="ExtraTree-splitter",
                persistence=True,persistence_type="memory"),

            getInputNumberDiv("Max Depth",
                              "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min samples split samples.",
                              "ExtraTree-max_depth","default None",1,0),
            
            getInputNumberDiv("Min Samples Split",
                              "The minimum number of samples required to split an internal node.",
                              "ExtraTree-min_samples_split",2,1,1),
            
            getInputNumberDiv("Min Samples Leaf",
                              "The minimum number of samples required to be at a leaf node",
                              "ExtraTree-min_samples_leaf",1,1,1),
            
            getInputNumberDiv("Min Impurity Decrease",
                              "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                              "ExtraTree-min_impurity_decrease","0.0",0.1,0),            
                    
            getInputNumberDiv("Max Leaf Nodes",
                              "Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
                              "ExtraTree-max_leaf_nodes","default None",1,0),


            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="ExtraTree-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("ExtraTree"),
                      id="ExtraTree-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="ExtraTree-collapse",is_open=False,),]





DecisionTree_content=[
    dbc.CardHeader(getAlgoHeader("DecisionTree")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
            
            dbc.Label("Criterion",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("The function to measure the quality of a split.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "Gini", "value": "gini"},
                    {"label": "Entropy", "value": "entropy"},                    
                ],value='gini',id="DecisionTree-criterion",
                persistence=True,persistence_type="memory"),
            

                        
            dbc.Label("Splitter",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("The strategy used to choose the split at each node.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "Random", "value": "random"},
                    {"label": "Best", "value": "best"},                    
                ],value='best',id="DecisionTree-splitter",
                persistence=True,persistence_type="memory"),

            getInputNumberDiv("Max Depth",
                              "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min samples split samples.",
                              "DecisionTree-max_depth","default None",1,0),
            
            getInputNumberDiv("Min Samples Split",
                              "The minimum number of samples required to split an internal node.",
                              "DecisionTree-min_samples_split",2,1,1),
            
            getInputNumberDiv("Min Samples Leaf",
                              "The minimum number of samples required to be at a leaf node",
                              "DecisionTree-min_samples_leaf",1,1,0),
            
            getInputNumberDiv("Min Impurity Decrease",
                              "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                              "DecisionTree-min_impurity_decrease","0.0",0.1,0),            
                    
            getInputNumberDiv("Max Leaf Nodes",
                              "Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
                              "DecisionTree-max_leaf_nodes","default None",1,0),

            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="DecisionTree-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("DecisionTree"),
                      id="DecisionTree-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="DecisionTree-collapse",is_open=False,),]



RandomForest_content=[
    dbc.CardHeader(getAlgoHeader("RandomForest")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
            getInputNumberDiv("N estimators",
                              "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min samples split samples.",
                              "RandomForest-n_estimators",100,1,10),
                        
            dbc.Label("Criterion",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("The function to measure the quality of a split.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "Gini", "value": "gini"},
                    {"label": "Entropy", "value": "entropy"},                    
                ],value='gini',id="RandomForest-criterion",
                persistence=True,persistence_type="memory"),
            
            getInputNumberDiv("Max Depth",
                              "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min samples split samples.",
                              "RandomForest-max_depth","default None",1,0),
            
            
            getInputNumberDiv("Min Samples Split",
                              "The minimum number of samples required to split an internal node.",
                              "RandomForest-min_samples_split",2,1,1),
            
            getInputNumberDiv("Min Samples Leaf",
                              "The minimum number of samples required to be at a leaf node",
                              "RandomForest-min_samples_leaf",1,1,1),
            
            getInputNumberDiv("Min Impurity Decrease",
                              "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                              "RandomForest-min_impurity_decrease","0.0",0.1,0),            
                    
            getInputNumberDiv("Max Leaf Nodes",
                              "Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
                              "RandomForest-max_leaf_nodes","default None",1,0),
            
                        
            dbc.Label("Bootstrap",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Whether samples are drawn with replacement. If False, sampling without replacement is performed.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=True,id="RandomForest-bootstrap",
                persistence=True,persistence_type="memory"),
            
            getInputNumberDiv("Max Samples",
                 "If bootstrap is True, the number of samples to draw from X to train each base estimator.",
                 "RandomForest-max_samples","default None",1,0),
                        
            
            dbc.Label("OOB Score",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Whether to use out-of-bag samples to estimate the generalization error. Only available if bootstrap=True.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=False,id="RandomForest-oob_score",
                persistence=True,persistence_type="memory"),
            
            dbc.Label("Warm Start",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble. ",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=False,id="RandomForest-warm_start",
                persistence=True,persistence_type="memory"),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="RandomForest-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("RandomForest"),
                      id="RandomForest-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="RandomForest-collapse",is_open=False,),]



LinearDiscriminantAnalysis_content=[
    dbc.CardHeader(getAlgoHeader("LinearDiscriminantAnalysis")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
                        
            dbc.Label("Solver",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            dbc.RadioItems(
                options=[
                    {"label": "SVD", "value": "svd"},
                    {"label": "LSQR", "value": "lsqr"},   
                    {"label": "Eigen", "value": "eigen"},               
                ],value='svd',id="LinearDiscriminantAnalysis-solver",
                persistence=True,persistence_type="memory"),
            
            dbc.Label("Shrinkage",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Note that shrinkage works only with ‘lsqr’ and ‘eigen’ solvers.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "None: no shrinkage ", "value": None},
                    {"label": "Auto: automatic shrinkage using the Ledoit-Wolf lemma", "value": "auto"},   
                ],value=None,id="LinearDiscriminantAnalysis-shrinkage",
                persistence=True,persistence_type="memory"),
            
            getInputNumberDiv("N Components",
                              "Number of components (default is min(n_classes - 1, n_features)) for dimensionality reduction.",
                              "LinearDiscriminantAnalysis-n_components","default None",1,0),
            
                     
            dbc.FormGroup([
                            dbc.Label("Tolerance",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("Absolute threshold for a singular value of X to be considered significant, used to estimate the rank of X. Dimensions whose singular values are non-significant are discarded. Only used if solver is ‘svd’.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.001", "value": 0.001},
                                    {"label": "0.0001", "value": 0.0001},
                                    {"label": "0.00001", "value": 0.00001},
                                ],value=0.0001, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="LinearDiscriminantAnalysis-tol",persistence=True,persistence_type="memory")

                         ]),
            
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="LinearDiscriminantAnalysis-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("LinearDiscriminantAnalysis"),
                      id="LinearDiscriminantAnalysis-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="LinearDiscriminantAnalysis-collapse",is_open=False,),]


LogisticRegression_content=[
    dbc.CardHeader(getAlgoHeader("LogisticRegression")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
            
            dbc.FormGroup([
                            dbc.Label("Specify the norm of the penalty",style={"font-weight": "bold","font-size": "18px"}),
                            html.Br(),
                            dbc.Button(html.I( className="fa fa-exclamation-triangle fa-sm"),id="LogisticRegression-pen-alert-btn",n_clicks=0,color="danger",style={"margin-top": "15px"}),
                            dbc.Alert(                                 
                                    html.Label(['Some penalties may not work with some solvers. Please check compatibility between the penalty and solver.',html.Br() ,html.Br() ," Please refer to the ",
                                    html.A('scikit-learn user guide', href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',target="_blank",style={"color": "black"}),
                               " for further details."],
                                   style={"text-align": "Justify"}),
                                
                                
                                      id="LogisticRegression-pen-alert-text",dismissable=True,is_open=False,color="info"),
                            
                            
                            
                            dcc.Dropdown(
                                options=[
                                    {"label": "none", "value": "None"},
                                    {"label": "l1", "value": "l1"},
                                    {"label": "l2", "value": "l2"},
                                    {"label": "elasticnet:  both L1 and L2 penalty terms are added.", "value": "elasticnet"},
                                ],value='l2', clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="LogisticRegression-penalty",persistence=True,persistence_type="memory")

                         ]),
                        
                        
            dbc.Label("Dual or Primal formulation",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Prefer dual=False when n_samples > n_features. Implemented for l2 penalty with liblinear solver.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=False,id="LogisticRegression-dual",
                persistence=True,persistence_type="memory"),
            
            dbc.FormGroup([
                            dbc.Label("Tolerance for stopping criteria",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.001", "value": 0.001},
                                    {"label": "0.0001", "value": 0.0001},
                                    {"label": "0.00001", "value": 0.00001},
                                ],value=0.0001, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="LogisticRegression-tol",persistence=True,persistence_type="memory")

                         ]),
                         
            getInputNumberDiv("C: Regularization parameter",
                              "inverse of regularization strength; must be a positive float.",
                              "LogisticRegression-C","1.0",1,0.1),
                        
            
            dbc.Label("Fit Intercept",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True ", "value": True},
                    {"label": "False", "value": False},   
                ],value=True,id="LogisticRegression-fit_intercept",
                persistence=True,persistence_type="memory"),
            
            
            dbc.FormGroup([
                            dbc.Label("Solver",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("Algorithm to use in the optimization problem.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "newton-cg", "value": "newton-cg"},
                                    {"label": "lbfgs", "value": "lbfgs"},
                                    {"label": "liblinear", "value": "liblinear"},
                                    {"label": "sag", "value": "sag"},
                                    {"label": "saga", "value": "saga"},
                                ],value="lbfgs", clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="LogisticRegression-solver",persistence=True,persistence_type="memory")

                         ]),
            

         
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="LogisticRegression-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("LogisticRegression"),
                      id="LogisticRegression-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="LogisticRegression-collapse",is_open=False,),]

GaussianProcess_content=[
    dbc.CardHeader(getAlgoHeader("GaussianProcess")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
                      
                                 
            getInputNumberDiv("N Restarts Optimizer",
                              "The number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood. Note that N Restarts Optimizer=0 implies that one run is performed.",
                              "GaussianProcess-n_restarts_optimizer",0,1,0),
                        
                                 
            getInputNumberDiv("Max Iter Predict",
                              "The maximum number of iterations in Newton’s method for approximating the posterior during predict. Smaller values will reduce computation time at the cost of worse results.",
                              "GaussianProcess-max_iter_predict",100,1,10),
                        
                        
            dbc.Label("Warm Start",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("If warm-starts are enabled, the solution of the last Newton iteration on the Laplace approximation of the posterior mode is used as initialization for the next call of _posterior_mode(). ",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=False,id="GaussianProcess-warm_start",
                persistence=True,persistence_type="memory"),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="GaussianProcess-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("GaussianProcess"),
                      id="GaussianProcess-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="GaussianProcess-collapse",is_open=False,),]


AdaBoost_content=[
    dbc.CardHeader(getAlgoHeader("AdaBoost")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
                      
                                 
            getInputNumberDiv("Number of Estimators",
                              "The maximum number of estimators at which boosting is terminated.",
                              "AdaBoost-n_estimators",50,1,10),
                        
                                 
            getInputNumberDiv("Learning Rate",
                              "Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier. There is a trade-off between the learning_rate and n_estimators parameters.",
                              "AdaBoost-learning_rate","1.0",1,0.1),
                        
                        
            dbc.Label("Algorithm",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            dbc.RadioItems(
                options=[
                    {"label": "SAMME", "value": "SAMME"},
                    {"label": "SAMME.R", "value": "SAMME.R"},   
                ],value="SAMME.R",id="AdaBoost-algorithm",
                persistence=True,persistence_type="memory"),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="AdaBoost-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("AdaBoost"),
                      id="AdaBoost-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="AdaBoost-collapse",is_open=False,),]



GradientBoosting_content=[
    dbc.CardHeader(getAlgoHeader("GradientBoosting")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
            dbc.Label("Loss",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("The loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "Deviance", "value": "deviance"},
                    {"label": "Exponential", "value": "exponential"},   
                ],value="deviance",id="GradientBoosting-loss",
                persistence=True,persistence_type="memory"),
                      
                     
                                 
            getInputNumberDiv("Learning Rate",
                              "Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.",
                              "GradientBoosting-learning_rate",0.1,0.1,0.1),
            
                                             
            getInputNumberDiv("Number of Estimators",
                              "The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.",
                              "GradientBoosting-n_estimators",100,1,10),
            
            dbc.FormGroup([
                            dbc.Label("Criterion",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("The function to measure the quality of a split.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "Friedman Mse: mean squared error with improvement score by Friedman", "value": "friedman_mse"},
                                    {"label": "Squared Error: mean squared error", "value": "squared_error"},
                                ],value="friedman_mse", clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="GradientBoosting-criterion",persistence=True,persistence_type="memory")

                         ]),
            

            
            getInputNumberDiv("Min Samples Split",
                              "The minimum number of samples required to split an internal node.",
                              "GradientBoosting-min_samples_split",2,1,1),
            
            getInputNumberDiv("Min Samples Leaf",
                              "The minimum number of samples required to be at a leaf node",
                              "GradientBoosting-min_samples_leaf",1,1,1),
            
                         
            getInputNumberDiv("Max Depth",
                              "The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. ",
                              "GradientBoosting-max_depth",3,1,1),
            
            getInputNumberDiv("Min Impurity Decrease",
                              "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                              "GradientBoosting-min_impurity_decrease","0.0",0.1,0),            
                    
            getInputNumberDiv("Max Leaf Nodes",
                              "Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
                              "GradientBoosting-max_leaf_nodes","default None",1,0),
                        
            dbc.Label("Warm Start",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble. ",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=False,id="GradientBoosting-warm_start",
                persistence=True,persistence_type="memory"),
            
            dbc.FormGroup([
                            dbc.Label("Tolerance for the early stopping.",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.001", "value": 0.001},
                                    {"label": "0.0001", "value": 0.0001},
                                    {"label": "0.00001", "value": 0.00001},
                                ],value=0.0001, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="GradientBoosting-tol",persistence=True,persistence_type="memory")

                         ]),
            
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="GradientBoosting-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("GradientBoosting"),
                      id="GradientBoosting-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="GradientBoosting-collapse",is_open=False,),]



Bagging_content=[
    dbc.CardHeader(getAlgoHeader("Bagging")),
    
    dbc.Collapse(
    dbc.CardBody(
        [            
                      
                                 
            getInputNumberDiv("Number of Estimators",
                              "The number of base estimators in the ensemble.",
                              "Bagging-n_estimators",10,1,5),
                        
                                 
            getInputNumberDiv("Max Samples",
                              "The number of samples to draw from X to train each base estimator.",
                              "Bagging-max_samples","1.0",1,1),
                        
                        
            getInputNumberDiv("Max Features",
                              "The number of features to draw from X to train each base estimator.",
                              "Bagging-max_features","1.0",1,1),
            
                        
            dbc.Label("Bootstrap",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Whether samples are drawn with replacement. If False, sampling without replacement is performed.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=True,id="Bagging-bootstrap",
                persistence=True,persistence_type="memory"),
            
            dbc.Label("Bootstrap features",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Whether features are drawn with replacement.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=False,id="Bagging-bootstrap_features",
                persistence=True,persistence_type="memory"),
            
            dbc.Label("OOB Score",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("Whether to use out-of-bag samples to estimate the generalization error. Only available if bootstrap=True.",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=False,id="Bagging-oob_score",
                persistence=True,persistence_type="memory"),
            
            dbc.Label("Warm Start",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
            html.Div(dbc.Label("When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble. ",style={"margin-top": "12px","font-size": "10px",})),
            dbc.RadioItems(
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},   
                ],value=False,id="Bagging-warm_start",
                persistence=True,persistence_type="memory"),
            

            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="Bagging-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("Bagging"),
                      id="Bagging-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="Bagging-collapse",is_open=False,),]




GaussianNB_content=[
    dbc.CardHeader(getAlgoHeader("GaussianNB")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
           
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="GaussianNB-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("GaussianNB"),
                      id="GaussianNB-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="GaussianNB-collapse",is_open=False,),
    ]

QuadraticDiscriminantAnalysis_content=[
    dbc.CardHeader(getAlgoHeader("QuadraticDiscriminantAnalysis")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            dbc.Label("Absolute Threshold",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("Absolute threshold for a singular value to be considered significant, used to estimate the rank of Xk where Xk is the centered matrix of samples in class k.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.001", "value": 0.001},
                                    {"label": "0.0001", "value": 0.0001},
                                    {"label": "0.00001", "value": 0.00001},
                                ],value=0.0001, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="QuadraticDiscriminantAnalysis-tol",persistence=True,persistence_type="memory")

                         ]),
           
            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="QuadraticDiscriminantAnalysis-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("QuadraticDiscriminantAnalysis"),
                      id="QuadraticDiscriminantAnalysis-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="QuadraticDiscriminantAnalysis-collapse",is_open=False,),
    ]

NearestCentroid_content=[
    dbc.CardHeader(getAlgoHeader("NearestCentroid")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            dbc.Label("Metric",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("The metric to use when calculating distance between instances in a feature array.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "cityblock", "value": "cityblock"},
                                    {"label": "cosine", "value":"cosine"},
                                    {"label": "euclidean", "value":"euclidean"},
                                    {"label": "l1", "value": "l1"},
                                    {"label": "l2", "value": "l2"},
                                    {"label": "manhattan", "value": "manhattan"},


                                ],value= "euclidean", clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="NearestCentroid-metric",persistence=True,persistence_type="memory"),
                            
                    html.Div(dbc.Label("Shrink Threshold",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"})),
                    html.Div(dbc.Label("Between 0 to 1",style={"margin-top": "12px","font-size": "10px",})),
                    dbc.Input(type="number",placeholder="default None", id="NearestCentroid-shrink_threshold",persistence=True,persistence_type="memory"),
                            

                         ]),
            

            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="NearestCentroid-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("NearestCentroid"),
                      id="NearestCentroid-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="NearestCentroid-collapse",is_open=False,),
    ]



SGD_content=[
    dbc.CardHeader(getAlgoHeader("SGD")),
    
    dbc.Collapse(
    dbc.CardBody(
        [
            dbc.FormGroup([
                            dbc.Label("Loss function to be used",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "Hinge", "value": "hinge"},
                                    {"label": "Log", "value":"log"},
                                    {"label": "Modified Huber", "value":"modified_huber"},
                                    {"label": "Squared Hinge", "value": "squared_hinge"},
                                    {"label": "Perceptron", "value": "perceptron"},
                                    {"label": "Squared Error", "value": "squared_error"},                                    {"label": "Squared Hinge", "value": "squared_hinge"},
                                    {"label": "Huber", "value": "huber"},
                                    {"label": "Epsilon Insensitive", "value": "epsilon_insensitive"},
                                    {"label": "Squared Epsilon Insensitive", "value": "squared_epsilon_insensitive"},


                                ],value= "hinge", clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SGD-loss",persistence=True,persistence_type="memory"),
                            
                            
                                                        
                            dbc.Label("Penalty to be used",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "L2", "value": "l2"},
                                    {"label": "L1", "value":"l1"},
                                    {"label": "Elasticnet", "value":"elasticnet"},
                                    
                                ],value= "l2", clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SGD-penalty",persistence=True,persistence_type="memory"),
                            
                            dbc.Label("Alpha",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("Constant that multiplies the regularization term. The higher the value, the stronger the regularization. ",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.001", "value": 0.001},
                                    {"label": "0.0001", "value": 0.0001},
                                    {"label": "0.00001", "value": 0.00001},
                                ],value=0.0001, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SGD-alpha",persistence=True,persistence_type="memory"),

                                        
                            
                            
                            dbc.Label("Fit Intercept",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.",style={"margin-top": "12px","font-size": "10px",})),
                            dbc.RadioItems(
                                options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False},   
                                ],value=True,id="SGD-fit_intercept",
                                persistence=True,persistence_type="memory"),
                            
                        
                            getInputNumberDiv("Max Iterations",
                              "The maximum number of passes over the training data (aka epochs). ",
                              "SGD-max_iter",1000,1,10),
                        
                            dbc.Label("Tol",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("The stopping criterion.",style={"margin-top": "12px","font-size": "10px",})),
                            dcc.Dropdown(
                                options=[
                                    {"label": "0.01", "value": 0.01},
                                    {"label": "0.001", "value": 0.001},
                                    {"label": "0.0001", "value": 0.0001},
                                    {"label": "0.00001", "value": 0.00001},
                                ],value=0.001, clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SGD-tol",persistence=True,persistence_type="memory"),

                            dbc.Label("Learning rate schedule",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            dcc.Dropdown(
                                options=[
                                    {"label": "Constant", "value": "constant"},
                                    {"label": "Optimal", "value":"optimal"},
                                    {"label": "Invscaling", "value":"invscaling"},
                                    {"label": "Adaptive", "value":"adaptive"},

                                    
                                ],value= "optimal", clearable=False,style={"font-size": "14px",'color': 'black'},
                                id="SGD-learning_rate",persistence=True,persistence_type="memory"),
                                                        
                            
                            getInputNumberDiv("Number of Iterations",
                              "Number of iterations with no improvement to wait before stopping fitting.",
                              "SGD-n_iter_no_change",5,5,2),
                                                        
                            dbc.Label("Warm Start",style={"margin-top": "15px","font-weight": "bold","font-size": "18px"}),
                            html.Div(dbc.Label("When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble. ",style={"margin-top": "12px","font-size": "10px",})),
                            dbc.RadioItems(
                                options=[
                                    {"label": "True", "value": True},
                                    {"label": "False", "value": False},   
                                ],value=False,id="SGD-warm_start",
                                persistence=True,persistence_type="memory"),
    

                         ]),
            

            dbc.Button(html.I( className="fa fa-info-circle fa-lg"),id="SGD-info-btn",n_clicks=0,style={"margin-top": "15px"}),
            dbc.Alert(infoText("SGD"),
                      id="SGD-info-text",dismissable=True,color="info",is_open=False,),
        ]),   
    id="SGD-collapse",is_open=False,),
    ]



    
   

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Card Layouts   
classAlgo_content =dbc.Card([
                 dbc.CardBody([
                        dbc.Row([
                                dbc.Col(dbc.Card(dummy_content, color="secondary", outline=True)),
                                dbc.Col(dbc.Card(SVM_content, color="secondary", outline=True)),
                                ],className="mb-4",),
                        

                        dbc.Row([

                                  dbc.Col(dbc.Card(KNN_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(ExtraTree_content, color="secondary", outline=True),),
                                ],className="mb-4",),
                        
                        
                        dbc.Row([

                                  dbc.Col(dbc.Card(DecisionTree_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(RandomForest_content, color="secondary", outline=True),),
                                ],className="mb-4",),
                        
                                                
                        
                        dbc.Row([

                                  dbc.Col(dbc.Card(LinearDiscriminantAnalysis_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(LogisticRegression_content, color="secondary", outline=True),),
                                ],className="mb-4",),
                        
                        
                        dbc.Row([

                                  dbc.Col(dbc.Card(GaussianProcess_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(AdaBoost_content, color="secondary", outline=True),),
                                ],className="mb-4",),
                        
                                                
                        dbc.Row([

                                  dbc.Col(dbc.Card(GradientBoosting_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(Bagging_content, color="secondary", outline=True),),
                                ],className="mb-4",),
                        
                                                
                        dbc.Row([

                                  dbc.Col(dbc.Card(GaussianNB_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(QuadraticDiscriminantAnalysis_content, color="secondary", outline=True),),
                                ],className="mb-4",),
                                                
                                                
                                                
                        dbc.Row([

                                  dbc.Col(dbc.Card(NearestCentroid_content, color="secondary", outline=True)),
                                  dbc.Col(dbc.Card(SGD_content, color="secondary", outline=True)),
                                ],className="mb-4",),
                        
                        
                            ]),
                 
                 

                ],className="mt-3",color="dark", outline=True)



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#list of all parameters of classssification algo
from UI.componentIDs import classification_Com_IDS
classification_Com_IDS=classification_Com_IDS.split(",")
classification_Com_IDS = [sub[1 : ] for sub in classification_Com_IDS]

#get all the algo names
global algoName
algoName=[]
for item in classification_Com_IDS:
    if "-" not in item and "_" not in item:
        algoName.append(item)

global paraname
paraname=[]

for item in classification_Com_IDS:
    #we dont need collapse and info state as of now
    if "collapse" in item or "info" in item or "alert" in item:
        continue
    #create a dict with algo name
    if item in algoName:
        continue
    #otherwise save its parameters
    else:
        paraname.append(item)

@app.callback(
    Output("classification_tab_para", 'data'),
    [Input("{}".format(_), 'value') for _ in algoName],
    [Input("{}".format(_), 'value') for _ in paraname]
    )
def get_classification_tab_input(*args): 
    
    #args are tupple of list containing boolean
    #([True], [True], [True], [True], [], [True], [], [], [], [], [])
    
    options=args[0:len(algoName)] #all active algo
    para_state=args[len(algoName):len(args)] #all par value

    para_indexer=0 #to track the current position of paraname and pasa state
    data={}         # to save all the data

    #first iterate through each algo state
    for i in range(0,len(options)):

        #check if that algo is active
        if len(options[i])>0:  
  
            data[algoName[i]]={}   #creat a empty list for that algo
            #iterate throgh the parastate and paraname simultaneously
            for j in range(para_indexer,len(para_state)):

                #sanity check to avoid wrong paraname and value pair formation
                if algoName[i] not in paraname[j]:
                    break    
                   
                #assign the para state to correspnding para name
                data[algoName[i]][paraname[j]]= para_state[j] 
                para_indexer+=1 
                
        else:
            #move the para state list indexer to the next algorithm 
            #parameters using paraname list
            for j in range(para_indexer,len(para_state)):
                if algoName[i] in paraname[j]: 
                    para_indexer+=1
                else: 
                    break
    return data




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#calllbacks of scaling info
genrateInfoCallback("Dummy Classifier")
genrateCollapseCallback("Dummy Classifier")

genrateInfoCallback("SVM")
genrateCollapseCallback("SVM")

genrateInfoCallback("KNN")
genrateCollapseCallback("KNN")

genrateInfoCallback("ExtraTree")
genrateCollapseCallback("ExtraTree")

genrateInfoCallback("DecisionTree")
genrateCollapseCallback("DecisionTree")


genrateInfoCallback("RandomForest")
genrateCollapseCallback("RandomForest")

genrateInfoCallback("LinearDiscriminantAnalysis")
genrateCollapseCallback("LinearDiscriminantAnalysis")


genrateInfoCallback("LogisticRegression")
genrateCollapseCallback("LogisticRegression")
genrateAlertCallback("LogisticRegression-pen")

genrateInfoCallback("GaussianProcess")
genrateCollapseCallback("GaussianProcess")

genrateInfoCallback("AdaBoost")
genrateCollapseCallback("AdaBoost")

genrateInfoCallback("GradientBoosting")
genrateCollapseCallback("GradientBoosting")


genrateInfoCallback("Bagging")
genrateCollapseCallback("Bagging")

genrateInfoCallback("GaussianNB")
genrateCollapseCallback("GaussianNB")


genrateInfoCallback("QuadraticDiscriminantAnalysis")
genrateCollapseCallback("QuadraticDiscriminantAnalysis")

genrateInfoCallback("NearestCentroid")
genrateCollapseCallback("NearestCentroid")

genrateInfoCallback("SGD")
genrateCollapseCallback("SGD")

