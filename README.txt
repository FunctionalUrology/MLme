
##########################
#### FILES CONTENT #######
##########################
-> scriptTemplate.py contains all the code to run the ML pipeline based on the user input(.pkl) file from the easyML.
-> inputParameters_hh_mm_ss.pkl contains all the input data from the app. 


##########################
####### HOW TO RUN #######
##########################

1. Open the terminal.
2. Navigate to the directory that contains the above-mentioned file. 
3. Execute the following code on your local machine or cluster:

	python scriptTemplate.py path/to/data.csv "sep" inputParameters_hh_mm_ss.pkl

				OR (if you want to run it as a background process)

	python scriptTemplate.py path/to/data.csv "sep" inputParameters_hh_mm_ss.pkl > bglog &

We highly recommend running this pipeline on High-Performance Computing (HPC) cluster.

##########################
######### OUTPUT #########
##########################


As an output, you will get a _trainedModels.pkl file. You can visualize the results using the visualisation tab in the app.
