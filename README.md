# Summary
<div style="text-align: right">  The github repo hosts a tool called "----tool name----". It offers data exploration, auto-ML, custom ML, and visualization features to enable users to use machine learning techniques in their research, irrespective of their technical skills. The tool's objective is to provide researchers with a comprehensive platform to explore their datasets, design tailor-fit ML pipelines, and visualize results in a clear and aesthetically pleasing manner. </div> 





# Installation

  - ### Prerequisite 
    
    To use ------------tool-name------------, you must have ```Python``` version ```3.9``` and ```pip``` installed
    and that they are accessible from the terminal.
    
  - ### Download sourcecode 
    Download the GitHub repository and unzip it.
    
  - ### Install dependencies  
    1. Open your terminal and change your current working directory to ------------tool-name------------ (e.g. ```cd path/to/------/```). 
    2. Please install the required packages using the following command: 
      
       ```pip install -r requirements.txt``` 
       
  - ### Launch  ------------tool-name------------
      
       ```python -m main``` 
       
  - ### Errors you may encounter


# Features

  - ### Data Exploration
    -  The Data Exploration feature enables users to upload datasets and analyze them with statistical visualizations, providing insights into data patterns, trends, and outliers for informed decisions in developing ML pipelines.  
    
    -  <i>Input</i>:  
       -  A .csv or .txt file with a row representing a sample and a column representing a feature. The first and last columns must contain the sample name and target classes, respectively, and the file must not have any NaN values.  
       -  When uploading a file, users must ensure that they select the correct separator using the ```Sep``` dropdown menu to avoid errors.
   
    -  <i>Output</i>:
       -  Users will be able to perform in-depth analysis on their datasets using statistical summary tables and five different plots, including density and correlation matrix plots. The analysis can be easily conducted by selecting the desired option from the ```Plot/Table Type``` dropdown menu.
       -  The users have the ability to download plots by clicking on the camera button that is provided on all the plots.
       
 - ### Auto ML
 
   - The Auto ML feature runs a default machine learning pipeline, allowing researchers to analyze their datasets without technical expertise. The pipeline includes preprocessing, feature selection, and training and evaluation of multiple classification models, including a dummy classifier. 
  
   -  <i>Input</i>:  
       -  A .csv or .txt file with a row representing a sample and a column representing a feature. The first and last columns must contain the sample name and target classes, respectively, and the file must not have any NaN values.  
       -  When uploading a file, users must ensure that they select the correct separator using the ```Sep``` dropdown menu to avoid errors.
       -  ```Variance Threshold```: The default ML pipeline incorporates a variance threshold feature that eliminates features with variance below the threshold specified by the user.
       -  ```No of Features to Select```: Specify the desired percentage of features to be selected from the original set by utilizing the feature selection step.


   -  <i>Output</i>: 
       -  A table with scores for 11 evaluation metrics for 6 ML algorithms: SVM, KNN, AdaBoost, GaussianNB, and Dummy classifier. 
       -  A table with selected features from the original set of features.
       -  Visualization of model performance through different plots such as spider plot and heatmap.
       -  Display of pipelines to see detailed steps and parameters of executed pipeline.
       -  A zip file available for download that contains a log file and all the results.pkl files. The user can then upload the results.pkl file to the visualization tab to interpret the results.
      
 - ### Custom ML
 
   -  It allows intermediate to advanced machine learning users to design a tailored machine learning pipeline as per the requirement. The user-friendly interface provides a simple toggle button to include or exclude steps/algorithms, allowing users to focus on selecting the most appropriate options for their dataset instead of programming.
   
   -  <i>Input</i>:  
       -  Users can easily choose which preprocessing steps (such as scaling, data resampling, and feature selection), classifier, model evaluation method, and evaluation metric score to include in their pipeline by clicking on a toggle button.
       -  The parameters of each individual algorithm can be customized by clicking on the ```Parameter``` button, which will provide a list of corresponding parameters that the user can adjust to their preference. If the user does not make any changes, default parameters will be used.
       - While preprocessing steps are optional, users are required to select at least one classifier, a model evaluation method, and a metric score.

   -  <i>Output</i>:  
       -  After selecting all the desired algorithms/steps, the user can click on the ```submit``` button to download a compressed zip file that includes files such as README.txt, inputParameter.pkl, and script.py.
       - The user can execute the designed pipeline on either their local machine or cluster by using the following command.  
         ```python script.py path/to/data.csv "seprator" inputParameter.pkl```
       - After the pipeline has been executed, it will generate a compressed zip file as output, which includes a log.txt and results.pkl file. The user can then upload the results.pkl file to the ```Visualization``` tab of ---tool name-- to interpret the results.
 
  - ### Visualization
 
   -  This feature enables users to effortlessly interpret and analyze their findings with the help of several interactive tables and plots.
   
   -  <i>Input</i>:  
       -  results.pkl file from ```Auto ML``` or ```Custom ML```.
       
   -  <i>Output</i>: 
       -  A range of tables and plots are available for comparative analysis of model performance. Users can customize and download all of the plots in high quality, making them suitable for publication.



# Errors you may encounter
  - ``` Following exception occurred: single positional indexer is out-of-bounds```
    - Possible solution: Ensure that you have selected the correct separator using the ```Sep``` dropdown menu.
