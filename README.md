# Summary
<div style="text-align: right">  The github repo hosts an app called "----app name----". It offers data exploration, auto-ML, custom ML, and visualization features to enable users to use machine learning techniques in their research, irrespective of their technical skills. The app's objective is to provide researchers with a comprehensive platform to explore their datasets, design tailor-fit ML pipelines, and visualize results in a clear and aesthetically pleasing manner. </div> 





# Installation

  - ### Prerequisite 
    
    To use ------------App-name------------, you must have ```Python``` version ```3.9``` and ```pip``` installed
    and that they are accessible from the terminal.
    
  - ### Download sourcecode 
    Download the GitHub repository and unzip it.
    
  - ### Install dependencies  
    1. Open your terminal and change your current working directory to ------------App-name------------ (e.g. ```cd path/to/------/```). 
    2. Please install the required packages using the following command: 
      
       ```pip install -r requirements.txt``` 
       
  - ### Launch  ------------App-name------------
      
       ```python -m main``` 
       
  - ### Errors you may encounter
    - ``` python setup.py bdist_wheel did not run successfully.```
      - Possible solution: You can simply ignore this error.

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
 
   - The Auto ML feature in an app runs a default machine learning pipeline, allowing researchers to analyze their datasets without technical expertise. The pipeline includes preprocessing, feature selection, and training and evaluation of multiple classification models, including a dummy classifier. 
  
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
      
   


 

