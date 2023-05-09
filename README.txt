##########################
#### CustomML Output #####
##########################

-> After selecting all the desired algorithms/steps, the user can click on the submit button to download a compressed zip file (userInputData.zip) that includes files such as README.txt, inputParameter.pkl, and scriptTemplate.py.

-> scriptTemplate.py contains all the code to run the ML pipeline based on the user input(.pkl) file from the easyML.

-> inputParameters_hh_mm_ss.pkl contains all the input data from the app. 

-> The user can run the designed pipeline on either their local machine or a cluster.

##########################
# Input file for pipeline 
##########################

A .csv or .txt file with a row representing a sample and a column representing a feature. The first and last columns must contain the sample name and target classes, respectively, and the file must not have any NaN values. Example input data.

##########################
####### HOW TO RUN #######
##########################

1. Open your terminal and change your directory to previosuly downloaded folder (userInputData) from customML. 

2. Run the follwoing command to execute the pipeline:

	 python scriptTemplate.py -i path/to/input.csv -p inputParameters.pkl -s tab -o .

				OR (if you want to run it as a background process)

	python scriptTemplate.py -i path/to/input.csv -p inputParameters.pkl -s tab -o . > bglog &


We highly recommend running this pipeline on High-Performance Computing (HPC) cluster.


##########################
#### Pipeline Tags #####
##########################

usage: scriptTemplate.py [-h] [-i INPUT] [-s SEPARATOR] [-p PARAMETERS] [-o OUTPUT]

##########################
#### Pipeline OUTPUT #####
##########################


-> After the pipeline has been executed, it will generate a compressed zip file as output, which includes a log.txt and results.pkl file. The user can then upload the results.pkl file to the Visualization tab of ---tool name-- to interpret the results.
