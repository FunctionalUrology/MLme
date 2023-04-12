#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:51:08 2022

@author: akshay
"""


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#IDs from diff tabs
scaling_Com_IDS=" MaxAbs Scaler, MinMax Scaler, Normalizer, PowerTransformer, QuantileTransformer, Robust Scaler, Standard Scaler, MinMax Scaler-feature_range, Normalizer-norm, PowerTransformer-method, PowerTransformer-standardize, QuantileTransformer-n_quantiles, QuantileTransformer-output_distribution, Robust Scaler-with_centering, Robust Scaler-with_scaling, Robust Scaler-quantile_range, Robust Scaler-unit_variance, Standard Scaler-with_mean, Standard Scaler-with_std"

overrsampling_Com_IDS=" ADASYN, SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, RandomOverSampler"

undersampling_Com_IDS=" AllKNN, AllKNN-collapse-button, AllKNN-collapse, AllKNN-sampling_strategy, AllKNN-n_neighbors, AllKNN-kind_sel, AllKNN-allow_minority, AllKNN-info-btn, AllKNN-info-text, ClusterCentroids, ClusterCentroids-collapse-button, ClusterCentroids-collapse, ClusterCentroids-sampling_strategy, ClusterCentroids-voting, ClusterCentroids-info-btn, ClusterCentroids-info-text, CondensedNearestNeighbour, CondensedNearestNeighbour-collapse-button, CondensedNearestNeighbour-collapse, CondensedNearestNeighbour-sampling_strategy, CondensedNearestNeighbour-n_neighbors, CondensedNearestNeighbour-n_seeds_S, CondensedNearestNeighbour-info-btn, CondensedNearestNeighbour-info-text, EditedNearestNeighbours, EditedNearestNeighbours-collapse-button, EditedNearestNeighbours-collapse, EditedNearestNeighbours-sampling_strategy, EditedNearestNeighbours-n_neighbors, EditedNearestNeighbours-kind_sel, EditedNearestNeighbours-info-btn, EditedNearestNeighbours-info-text, InstanceHardnessThreshold, InstanceHardnessThreshold-collapse-button, InstanceHardnessThreshold-collapse, InstanceHardnessThreshold-sampling_strategy, InstanceHardnessThreshold-cv, InstanceHardnessThreshold-info-btn, InstanceHardnessThreshold-info-text, NearMiss, NearMiss-collapse-button, NearMiss-collapse, NearMiss-sampling_strategy, NearMiss-n_neighbors, NearMiss-info-btn, NearMiss-info-text, NeighbourhoodCleaningRule, NeighbourhoodCleaningRule-collapse-button, NeighbourhoodCleaningRule-collapse, NeighbourhoodCleaningRule-sampling_strategy, NeighbourhoodCleaningRule-n_neighbors, NeighbourhoodCleaningRule-kind_sel, NeighbourhoodCleaningRule-threshold_cleaning, NeighbourhoodCleaningRule-info-btn, NeighbourhoodCleaningRule-info-text, OneSidedSelection, OneSidedSelection-collapse-button, OneSidedSelection-collapse, OneSidedSelection-sampling_strategy, OneSidedSelection-n_neighbors, OneSidedSelection-n_seeds_S, OneSidedSelection-info-btn, OneSidedSelection-info-text, RandomUnderSampler, RandomUnderSampler-collapse-button, RandomUnderSampler-collapse, RandomUnderSampler-sampling_strategy, RandomUnderSampler-replacement, RandomUnderSampler-info-btn, RandomUnderSampler-info-text, RepeatedEditedNearestNeighbours, RepeatedEditedNearestNeighbours-collapse-button, RepeatedEditedNearestNeighbours-collapse, RepeatedEditedNearestNeighbours-sampling_strategy, RepeatedEditedNearestNeighbours-n_neighbors, RepeatedEditedNearestNeighbours-kind_sel, RepeatedEditedNearestNeighbours-info-btn, RepeatedEditedNearestNeighbours-info-text, TomekLinks, TomekLinks-collapse-button, TomekLinks-collapse, TomekLinks-sampling_strategy, TomekLinks-info-btn, TomekLinks-info-text"


featSel_Com_IDS=" RFECV, RFECV-collapse-button, RFECV-collapse, RFECV-estimator, RFECV-step, RFECV-min_features_to_select, RFECV-info-btn, RFECV-info-text, SelectFdr, SelectFdr-collapse-button, SelectFdr-collapse, SelectFdr-score_func, SelectFdr-alpha, SelectFdr-info-btn, SelectFdr-info-text, SelectFpr, SelectFpr-collapse-button, SelectFpr-collapse, SelectFpr-score_func, SelectFpr-alpha, SelectFpr-info-btn, SelectFpr-info-text, SelectFromModel, SelectFromModel-collapse-button, SelectFromModel-collapse, SelectFromModel-estimator, SelectFromModel-max_features, SelectFromModel-info-btn, SelectFromModel-info-text, SelectFwe, SelectFwe-collapse-button, SelectFwe-collapse, SelectFwe-score_func, SelectFwe-alpha, SelectFwe-info-btn, SelectFwe-info-text, SelectKBest, SelectKBest-collapse-button, SelectKBest-collapse, SelectKBest-score_func, SelectKBest-k, SelectKBest-info-btn, SelectKBest-info-text, SequentialFeatureSelector, SequentialFeatureSelector-collapse-button, SequentialFeatureSelector-collapse, SequentialFeatureSelector-estimator, SequentialFeatureSelector-n_features_to_select, SequentialFeatureSelector-direction, SequentialFeatureSelector-info-btn, SequentialFeatureSelector-info-text, SelectPercentile, SelectPercentile-collapse-button, SelectPercentile-collapse, SelectPercentile-score_func, SelectPercentile-percentile, SelectPercentile-info-btn, SelectPercentile-info-text, VarianceThreshold, VarianceThreshold-collapse-button, VarianceThreshold-collapse, VarianceThreshold-threshold, VarianceThreshold-info-btn, VarianceThreshold-info-text"

classification_Com_IDS=" Dummy Classifier, Dummy Classifier-collapse-button, Dummy Classifier-collapse, Dummy Classifier-strategy, Dummy Classifier-info-btn, Dummy Classifier-info-text, SVM, SVM-collapse-button, SVM-collapse, SVM-C, SVM-kernel, SVM-degree, SVM-gamma, SVM-tol, SVM-info-btn, SVM-info-text, KNN, KNN-collapse-button, KNN-collapse, KNN-n_neighbors, KNN-weights, KNN-algorithm, KNN-leaf_size, KNN-p, KNN-info-btn, KNN-info-text, ExtraTree, ExtraTree-collapse-button, ExtraTree-collapse, ExtraTree-criterion, ExtraTree-splitter, ExtraTree-max_depth, ExtraTree-min_samples_split, ExtraTree-min_samples_leaf, ExtraTree-min_impurity_decrease, ExtraTree-max_leaf_nodes, ExtraTree-info-btn, ExtraTree-info-text, DecisionTree, DecisionTree-collapse-button, DecisionTree-collapse, DecisionTree-criterion, DecisionTree-splitter, DecisionTree-max_depth, DecisionTree-min_samples_split, DecisionTree-min_samples_leaf, DecisionTree-min_impurity_decrease, DecisionTree-max_leaf_nodes, DecisionTree-info-btn, DecisionTree-info-text, RandomForest, RandomForest-collapse-button, RandomForest-collapse, RandomForest-n_estimators, RandomForest-criterion, RandomForest-max_depth, RandomForest-min_samples_split, RandomForest-min_samples_leaf, RandomForest-min_impurity_decrease, RandomForest-max_leaf_nodes, RandomForest-bootstrap, RandomForest-max_samples, RandomForest-oob_score, RandomForest-warm_start, RandomForest-info-btn, RandomForest-info-text, LinearDiscriminantAnalysis, LinearDiscriminantAnalysis-collapse-button, LinearDiscriminantAnalysis-collapse, LinearDiscriminantAnalysis-solver, LinearDiscriminantAnalysis-shrinkage, LinearDiscriminantAnalysis-n_components, LinearDiscriminantAnalysis-tol, LinearDiscriminantAnalysis-info-btn, LinearDiscriminantAnalysis-info-text, LogisticRegression, LogisticRegression-collapse-button, LogisticRegression-collapse, LogisticRegression-pen-alert-btn, LogisticRegression-pen-alert-text, LogisticRegression-penalty, LogisticRegression-dual, LogisticRegression-tol, LogisticRegression-C, LogisticRegression-fit_intercept, LogisticRegression-solver, LogisticRegression-info-btn, LogisticRegression-info-text, GaussianProcess, GaussianProcess-collapse-button, GaussianProcess-collapse, GaussianProcess-n_restarts_optimizer, GaussianProcess-max_iter_predict, GaussianProcess-warm_start, GaussianProcess-info-btn, GaussianProcess-info-text, AdaBoost, AdaBoost-collapse-button, AdaBoost-collapse, AdaBoost-n_estimators, AdaBoost-learning_rate, AdaBoost-algorithm, AdaBoost-info-btn, AdaBoost-info-text, GradientBoosting, GradientBoosting-collapse-button, GradientBoosting-collapse, GradientBoosting-loss, GradientBoosting-learning_rate, GradientBoosting-n_estimators, GradientBoosting-criterion, GradientBoosting-min_samples_split, GradientBoosting-min_samples_leaf, GradientBoosting-max_depth, GradientBoosting-min_impurity_decrease, GradientBoosting-max_leaf_nodes, GradientBoosting-warm_start, GradientBoosting-tol, GradientBoosting-info-btn, GradientBoosting-info-text, Bagging, Bagging-collapse-button, Bagging-collapse, Bagging-n_estimators, Bagging-max_samples, Bagging-max_features, Bagging-bootstrap, Bagging-bootstrap_features, Bagging-oob_score, Bagging-warm_start, Bagging-info-btn, Bagging-info-text, GaussianNB, GaussianNB-collapse-button, GaussianNB-collapse, GaussianNB-info-btn, GaussianNB-info-text, QuadraticDiscriminantAnalysis, QuadraticDiscriminantAnalysis-collapse-button, QuadraticDiscriminantAnalysis-collapse, QuadraticDiscriminantAnalysis-tol, QuadraticDiscriminantAnalysis-info-btn, QuadraticDiscriminantAnalysis-info-text, NearestCentroid, NearestCentroid-collapse-button, NearestCentroid-collapse, NearestCentroid-metric, NearestCentroid-shrink_threshold, NearestCentroid-info-btn, NearestCentroid-info-text, SGD, SGD-collapse-button, SGD-collapse, SGD-loss, SGD-penalty, SGD-alpha, SGD-fit_intercept, SGD-max_iter, SGD-tol, SGD-learning_rate, SGD-n_iter_no_change, SGD-warm_start, SGD-info-btn, SGD-info-text"

modelEval_Com_IDS=" KFold, KFold-collapse-button, KFold-collapse, KFold-n_splits, KFold-shuffle, KFold-info-btn, KFold-info-text, StratifiedKFold, StratifiedKFold-collapse-button, StratifiedKFold-collapse, StratifiedKFold-n_splits, StratifiedKFold-shuffle, StratifiedKFold-info-btn, StratifiedKFold-info-text, RepeatedKFold, RepeatedKFold-collapse-button, RepeatedKFold-collapse, RepeatedKFold-n_splits, RepeatedKFold-n_repeats, RepeatedKFold-info-btn, RepeatedKFold-info-text, RepeatedStratifiedKFold, RepeatedStratifiedKFold-collapse-button, RepeatedStratifiedKFold-collapse, RepeatedStratifiedKFold-n_splits, RepeatedStratifiedKFold-n_repeats, RepeatedStratifiedKFold-info-btn, RepeatedStratifiedKFold-info-text, LeaveOneOut, LeaveOneOut-collapse-button, LeaveOneOut-collapse, LeaveOneOut-info-btn, LeaveOneOut-info-text, LeavePOut, LeavePOut-p, LeavePOut-collapse-button, LeavePOut-collapse, LeavePOut-info-btn, LeavePOut-info-text, ShuffleSplit, ShuffleSplit-collapse-button, ShuffleSplit-collapse, ShuffleSplit-test_size, ShuffleSplit-train_size, ShuffleSplit-info-btn, ShuffleSplit-info-text, StratifiedShuffleSplit, StratifiedShuffleSplit-collapse-button, StratifiedShuffleSplit-collapse, StratifiedShuffleSplit-test_size, StratifiedShuffleSplit-train_size, StratifiedShuffleSplit-info-btn, StratifiedShuffleSplit-info-text, NestedCV, NestedCV-collapse-button, NestedCV-collapse, NestedCV-n_splits, NestedCV-shuffle, NestedCV-info-btn, NestedCV-info-text"

# =============================================================================
# Iniate models
# =============================================================================
 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Diff scaling algo names
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,Normalizer,\
    PowerTransformer,QuantileTransformer,RobustScaler,StandardScaler
    
scaling_models=[MaxAbsScaler(),MinMaxScaler(),Normalizer(),\
    PowerTransformer(),QuantileTransformer(),RobustScaler(),StandardScaler()]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Diff over sampling algo names 
from imblearn.over_sampling import *
overSamp_models=[ADASYN(),SMOTE(),BorderlineSMOTE(),KMeansSMOTE(),SVMSMOTE(),RandomOverSampler()]
   

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Diff undersampling algo names
from imblearn.under_sampling import *
 
underSamp_models=[AllKNN(),
 ClusterCentroids(),
 CondensedNearestNeighbour(),
 EditedNearestNeighbours(),
 InstanceHardnessThreshold(),
 NearMiss(),
 NeighbourhoodCleaningRule(),
 OneSidedSelection(),
 RandomUnderSampler(),
 RepeatedEditedNearestNeighbours(),
 TomekLinks()]


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Diff classifcation model names
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier 
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier,GradientBoostingClassifier

# =============================================================================
# #get all classification algorithms
# from sklearn.utils import all_estimators
# estimators = all_estimators(type_filter='classifier')
# 
# all_clfs = []
# for name, ClassifierClass in estimators:
#     print('Appending', name)
#     try:
#         clf = ClassifierClass()
#         all_clfs.append(clf)
#     except Exception as e:
#         print('Unable to import', name)
#         print(e)
# =============================================================================
        
classification_models=[
 DummyClassifier(),
 SVC(probability=True),

 KNeighborsClassifier(),
 ExtraTreeClassifier(),

 DecisionTreeClassifier(),
 RandomForestClassifier(),

 LinearDiscriminantAnalysis(),
 LogisticRegression(),

 GaussianProcessClassifier(),
 AdaBoostClassifier(),

 GradientBoostingClassifier(),
 BaggingClassifier(),

 GaussianNB(),
 QuadraticDiscriminantAnalysis(),

 NearestCentroid(),
 SGDClassifier()]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Diff featureSel algo names
from sklearn.feature_selection import RFECV,SelectFdr,SelectFpr,\
    SelectFromModel,SelectFwe,SelectKBest,SequentialFeatureSelector,SelectPercentile,VarianceThreshold
 
featSel_models=[RFECV(SVC(probability=True)),SelectFdr(),SelectFpr(),\
    SelectFromModel( SVC(probability=True)),SelectFwe(),SelectKBest(),SequentialFeatureSelector( SVC(probability=True)),SelectPercentile(),VarianceThreshold()]

featSel_est={
 "SVM":SVC(probability=True,kernel="linear"),
 "LogisticRegression":LogisticRegression(),
 "ExtraTreeClassifier":ExtraTreeClassifier(),
 "DecisionTreeClassifier":DecisionTreeClassifier(),
 "LinearDiscriminantAnalysis":LinearDiscriminantAnalysis()
}




# =============================================================================
# Info Button text
# =============================================================================


# =============================================================================
# def getAlgoName(classification_Com_IDS):
#     classification_Com_IDS=classification_Com_IDS.split(",")
#     classification_Com_IDS = [sub[1 : ] for sub in classification_Com_IDS]
#     
#     #get all the algo names
#     global algoName
#     algoName=[]
#     for item in classification_Com_IDS:
#         if "-" not in item and "_" not in item:
#             algoName.append(item)
#     return algoName
# 
# allAlgoNames=getAlgoName(scaling_Com_IDS)+getAlgoName(scaling_Com_IDS)+getAlgoName(overrsampling_Com_IDS)\
#     +getAlgoName(undersampling_Com_IDS)+getAlgoName(featSel_Com_IDS)\
#         +getAlgoName(classification_Com_IDS) +getAlgoName(modelEval_Com_IDS)
#         
# allIds={k:["",""] for k in allAlgoNames}
# =============================================================================

from dash import html

allIds={'MaxAbs Scaler': ['This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.This scaler can also be applied to sparse CSR or CSC matrices.', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html'],
 'MinMax Scaler': ['This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html'],
 'Normalizer': ['Normalize samples individually to unit norm. Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1, l2 or inf) equals one.', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html'],
 'PowerTransformer': ['Apply a power transform featurewise to make data more Gaussian-like. Power transforms are a family of parametric, monotonic transformations that are applied to make data more Gaussian-like. This is useful for modeling issues related to heteroscedasticity (non-constant variance), or other situations where normality is desired.', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html'],
 'QuantileTransformer': ['Transform features using quantiles information. This method transforms the features to follow a uniform or a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is therefore a robust preprocessing scheme.', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html'],
 'Robust Scaler': ['Scale features using statistics that are robust to outliers. This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html'],
 'Standard Scaler': ['Standardize features by removing the mean and scaling to unit variance. Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. ', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html'],
 
 'ADASYN': ['Oversample using Adaptive Synthetic (ADASYN) algorithm. This method is similar to SMOTE but it generates different number of samples depending on an estimate of the local distribution of the class to be oversampled.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html'],
 'SMOTE': ['SMOTE is a Synthetic Minority Over-sampling Technique.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html'],
 'BorderlineSMOTE': ['Over-sampling using Borderline SMOTE. This algorithm is a variant of the original SMOTE algorithm. Borderline samples will be detected and used to generate new synthetic samples.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html'],
 'KMeansSMOTE': ['Apply a KMeans clustering before to over-sample using SMOTE.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.KMeansSMOTE.html'],
 'SVMSMOTE': ['Over-sampling using SVM-SMOTE. Variant of SMOTE algorithm which use an SVM algorithm to detect sample to use for generating new synthetic samples.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SVMSMOTE.html'],
 'RandomOverSampler': ['Class to perform random over-sampling. Object to over-sample the minority class(es) by picking samples at random with replacement. The bootstrap can be generated in a smoothed manner.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html'],
 
 'AllKNN': ['Undersample based on the AllKNN method. This method will apply ENN several time and will vary the number of nearest neighbours.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.AllKNN.html'],
 'ClusterCentroids': ['Undersample by generating centroids based on clustering methods. Method that under samples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm. This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class and using the coordinates of the N cluster centroids as the new majority samples.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html'],
 'CondensedNearestNeighbour': ['Undersample based on the condensed nearest neighbour method.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.CondensedNearestNeighbour.html'],
 'EditedNearestNeighbours': ['Undersample based on the edited nearest neighbour method. This method will clean the database by removing samples close to the decision boundary.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html'],
 'InstanceHardnessThreshold': ['Undersample based on the instance hardness threshold.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.InstanceHardnessThreshold.html'],
 'NearMiss': ['Class to perform under-sampling based on NearMiss methods.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NearMiss.html'],
 'NeighbourhoodCleaningRule': ['Undersample based on the neighbourhood cleaning rule. This class uses ENN and a k-NN to remove noisy samples from the datasets.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NeighbourhoodCleaningRule.html'],
 'OneSidedSelection': ['Class to perform under-sampling based on one-sided selection method.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.OneSidedSelection.html'],
 'RandomUnderSampler': ['Under-sample the majority class(es) by randomly picking samples with or without replacement.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html'],
 'RepeatedEditedNearestNeighbours': ['Undersample based on the repeated edited nearest neighbour method. This method will repeat several time the ENN algorithm', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RepeatedEditedNearestNeighbours.html'],
 'TomekLinks': ['Under-sampling by removing Tomek’s links.', 'https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.TomekLinks.html'],
 
 'RFECV': ['Recursive feature elimination with cross-validation to select the number of features.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html'],
 'SelectFdr': ['Filter: Select the p-values for an estimated false discovery rate. This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html'],
 'SelectFpr': ['Filter: Select the pvalues below alpha based on a FPR test. FPR test stands for False Positive Rate test. It controls the total amount of false detections.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html'],
 'SelectFromModel': ['Meta-transformer for selecting features based on importance weights.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html'],
 'SelectFwe': ['Filter: Select the p-values corresponding to Family-wise error rate.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html'],
 'SelectKBest': ['Select features according to the k highest scores.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html'],
 'SequentialFeatureSelector': ['Sequential Feature Selector adds (forward selection) or removes (backward selection) features to form a feature subset in a greedy fashion. At each stage, this estimator chooses the best feature to add or remove based on the cross-validation score of an estimator.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html'],
 'SelectPercentile': ['Select features according to a percentile of the highest scores.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html'],
 'VarianceThreshold': ['Feature selector that removes all low-variance features.', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html'],
 
 'Dummy Classifier': ['DummyClassifier makes predictions that ignore the input features. This classifier serves as a simple baseline to compare against other more complex classifiers.', 'https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html'],
 'SVM': ['C-Support Vector Classification.', 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'],
 'KNN': ['Classifier implementing the k-nearest neighbors vote.', 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'],
 'ExtraTree': ['An extremely randomized tree classifier. Extra-trees differ from classic decision trees in the way they are built. When looking for the best split to separate the samples of a node into two groups, random splits are drawn for each of the max_features randomly selected features and the best split among those is chosen. When max_features is set 1, this amounts to building a totally random decision tree. Warning: Extra-trees should only be used within ensemble methods.', 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html'],
 'DecisionTree': ['A decision tree classifier. Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.', 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html'],
 'RandomForest': ['A random forest classifier. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'],
 'LinearDiscriminantAnalysis': ['Linear Discriminant Analysis. A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix.', 'https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html'],
 'LogisticRegression': ['Logistic Regression (aka logit, MaxEnt) classifier. This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that regularization is applied by default.', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'],
 'GaussianProcess': ['Gaussian process classification (GPC) based on Laplace approximation. Internally, the Laplace approximation is used for approximating the non-Gaussian posterior by a Gaussian.', 'https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html'],
 'AdaBoost': ['An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html'],
 'GradientBoosting': ['Gradient Boosting for classification. GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the loss function, e.g. binary or multiclass log loss. Binary classification is a special case where only a single regression tree is induced.', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'],
 'Bagging': ['A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html'],
 'GaussianNB': ['Gaussian Naive Bayes (GaussianNB).', 'https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html'],
 'QuadraticDiscriminantAnalysis': ['Quadratic Discriminant Analysis. A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class.', 'https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html'],
 'NearestCentroid': ['Nearest centroid classifier. Each class is represented by its centroid, with test samples classified to the class with the nearest centroid.', 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html'],
 'SGD': ['Linear classifiers (SVM, logistic regression, etc.) with SGD training. This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning via the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance.', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html'],
 
 'KFold': ['K-Folds cross-validator. Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default). Each fold is then used once as a validation while the k - 1 remaining folds form the training set.', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html'],
 'StratifiedKFold': ['Stratified K-Folds cross-validator. Provides train/test indices to split data in train/test sets. This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html'],
 'RepeatedKFold': ['Repeated K-Fold cross validator. Repeats K-Fold n times with different randomization in each repetition.', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html'],
 'RepeatedStratifiedKFold': ['Repeated Stratified K-Fold cross validator. Repeats Stratified K-Fold n times with different randomization in each repetition.', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html'],
 'LeaveOneOut': ['Leave-One-Out cross-validator. Provides train/test indices to split data in train/test sets. Each sample is used once as a test set (singleton) while the remaining samples form the training set. Note: LeaveOneOut() is equivalent to KFold(n_splits=n) and LeavePOut(p=1) where n is the number of samples. Due to the high number of test sets (which is the same as the number of samples) this cross-validation method can be very costly. For large datasets one should favor KFold, ShuffleSplit or StratifiedKFold.', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html'],
 'LeavePOut': ['Leave-P-Out cross-validator. Provides train/test indices to split data in train/test sets. This results in testing on all distinct samples of size p, while the remaining n - p samples form the training set in each iteration. Note: LeavePOut(p) is NOT equivalent to KFold(n_splits=n_samples // p) which creates non-overlapping test sets. Due to the high number of iterations which grows combinatorically with the number of samples this cross-validation method can be very costly. For large datasets one should favor KFold, StratifiedKFold or ShuffleSplit.', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html'],
 'ShuffleSplit': ['Random permutation cross-validator. Yields indices to split data into training and test sets. Note: contrary to other cross-validation strategies, random splits do not guarantee that all folds will be different, although this is still very likely for sizeable datasets.', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html'],
 'StratifiedShuffleSplit': ['Stratified ShuffleSplit cross-validator. Provides train/test indices to split data in train/test sets. This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class. Note: like the ShuffleSplit strategy, stratified random splits do not guarantee that all folds will be different, although this is still very likely for sizeable datasets.', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html'],
 'NestedCV': ['Nested cross-validation (CV) is often used to train a model in which hyperparameters also need to be optimized. Nested CV estimates the generalization error of the underlying model and its (hyper)parameter search. Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score.', 'https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html']}

