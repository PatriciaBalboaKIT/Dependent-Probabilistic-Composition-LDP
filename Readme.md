
# Dependent Probabilistic Composition in LDP
### with Applications to Streaming Length Protection

This repository should contain:
- Long_version.pdf is the extended version of the submission that contains all the formal proofs presented in the paper.
- Code directory contains the source code used for the utility experiments the code necessary to run the LenPro mechanism for trajectory pretecion,
 as well as the code to run the baseline comparion IndReach and evaluate the results acording to the utility metrics established in the submission.

## Python requirements
Can be found in the file requirements.txt

## Pre-processing
The folder {data_type}_Databases/Pre_processing_Graph_Extraction contains the program road_graph.py.

Change the parameters, center_point and radius to ajust it to your desired geographical area and change time_interval according to the time between events in you database.

## Data Format
Each mechanism inputs trajectory datasets in file.csv such that each row is a trajectory where the first column contains the intial timestamp and the second a list of nodes from OpenStreetMap corresponding to street intersections.

An example dataset corresponding to the Porto map with 20 trajectories is provided in Code/Porto_Databases: test_Porto.csv

Code/Porto_Databases/Map_matching_Porto.py allows to obtain the map-matched database in the correct format from train.csv (https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data)

Code/Porto_Databases/fix_date_database_generator.py provides a subdatabase with N trajectories following an inverse exponential distribution biased towards the max_len. 

Code/Porto_Databases/data_time_intervals.py provide the time_intervals.csv files corresponding to the original datasets, required to run the utility metrics.

## Mechanisms 
The code to run the tree mechanism (LenPro,AdaptIndReach and IndReach) can be found in their corresponding folders.

#### Previous Steps:
Before runing any experiments we need to make sure that:
 -- The Map has been saved in {data_type}_Databases/Pre_processing_Graph_Extraction/saved_road_data.pkl 
    (This can be achieve executing road_graph.py)
 -- The database have being generated and can be found in the correct format in {data_type}_Databases

#### Execution
To run one experiment (consisting in 10 iterations) one needs to run main.py in the correct folder and input in the terminal de desired parameters.

Example:
> python MetricLenPro/main.py --data-type Porto --max-len 20 --N 5000 --eps_s 10 --eps_l 0.025 --interval 15

## Utility Metrics

The directory UtilityMetrics contains:

- utility_main2.py: the function to compute the macro metrics analyzed in this work
- grid.py : the function to customize the grid of the map
- {mechanism_name}_utility.py : functions to addapt the utility computation to each mechanism results format
- main_{mechanism_name}.py that allow to run the utility metrics in each of the tested mechanisms

Example:
> python UtilityMetrics/main_MLenPro.py --data-type Porto --max-len 20 --N 5000 --eps_s 10 --eps_l 0.5
