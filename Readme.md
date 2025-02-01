#Readme
# Dependent Probabilistic Composition in LDP
### with Applications to Streaming Length Protection

This repository should contain the code necessary to run the LenPro mechanism for trajectory pretecion, as well as the code to run the baseline comparion IndReach and evaluate the results acording to the utility metrics established in the submission.
## Pre-processing
The folder {data_type}_Databases/Pre_processing_Graph_Extraction contains the program road_graph.py.

Change the parameters, center_point and radius to ajust it to your desired geographical area and change time_interval according to the time between events in you database.

## Data Format
Each mechanism inputs trajectory datasets in file.csv such that each row is a trajectory where the first column contains the intial timestamp and the second a list of nodes from OpenStreetMap corresponding to street intersections.

## Mechanisms 
The code to run the tree mechanism (LenPro,AdaptIndReach and IndReach) can be found in their corresponding folders.

#### Previous Steps:
Before runing any experiments we need to make sure that:
 -- The Map has being downloaded and saved in {data_type}_Databases/Pre_processing_Graph_Extraction
 -- The database have being generated and can be found in the correct format in {data_type}_Databases

#### Execution
To run one experiment (consisting in 10 iterations) one needs to run main.py in the correct folder and input in the terminal de desired parameters.

Example:
> python MetricLenPro/main.py --data-type Porto --max-len 20 --N 5000 --eps_s 10 --eps_l 0.025 --interval 15