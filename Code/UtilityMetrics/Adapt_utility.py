import csv
import os
import numpy as np
import pickle
import networkx as nx
from ast import literal_eval
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from functools import lru_cache
from itertools import product
from multiprocessing import Pool
import pandas as pd
import datetime
import sys
#FUNCTIONS
# Route to the directory where the function is
directory_path = "UtilityMetrics"

# Add directory to sys.path
sys.path.append(directory_path)

# Now you can import the function
from grid import node_to_grid
from utility_main2 import *


def adapt_utility(eps_s, N, max_len, data_type):
    # Route of the files
    ORIGINAL_TRAJECTORIES_FILE = f"Results_{data_type}_Adapt_epsS{eps_s}_N{N}_{max_len}/original_data.csv"
    ANON_TRAJECTORIES_TEMPLATE = f"Results_{data_type}_Adapt_epsS{eps_s}_N{N}_{max_len}/Trajectories/anon{{}}.csv"
    OUTPUT_FILE = f"Utility_{data_type}_Adapt_{data_type}_{max_len}_epsS{eps_s}.csv"

    ##### GRAPH ######################################################

    GRAPH_FILE = f"{data_type}_Databases/Pre_processing_Graph_Extraction/saved_road_data.pkl"
    with open(GRAPH_FILE, "rb") as file:
        data = pickle.load(file)
    G = data["G"]
    LOCATIONS = set(G.nodes)
    NUM_LOCATIONS = len(LOCATIONS)

    ##### TIME INTERVALS ####################################################

    # File of time intervals
    TIME_INTERVALS_FILE = f"{data_type}_time_intervals_{max_len}.csv"
    # Load time intervals
    time_intervals_df = pd.read_csv(TIME_INTERVALS_FILE)
    time_intervals_dict = {(row['start'], row['end']): 0 for _, row in time_intervals_df.iterrows()}

    ##### ORIGINAL TRAJECTORIES ##############################################
    # Load original trajectories
    original_trajectories = pd.read_csv(ORIGINAL_TRAJECTORIES_FILE, header=None).values.tolist()

    ##### METRICS COMPARING ANON VS ORIGINAL #####################################
    jsd_values = []
    counting_query_error_values = []
    trip_error_values = []
    length_error_values = []
    abs_length_error_values = []
  

    # Iterate over 10 experiments
    for i in range(10):
        anon_trajectories_file = ANON_TRAJECTORIES_TEMPLATE.format(i)
        anon_trajectories = pd.read_csv(anon_trajectories_file, header=None).values.tolist()


        # Compute metrics
        counting_error = calculate_counting_query_error_vectorized(original_trajectories, anon_trajectories,LOCATIONS,time_intervals_dict)
        trip_error = calculate_trip_error(G, original_trajectories, anon_trajectories)
        density_error = calculate_density_error(original_trajectories, anon_trajectories,LOCATIONS)
        absolute_length_error = length_error(original_trajectories, anon_trajectories)

        # Add metrics to the lists
        counting_query_error_values.append(counting_error)
        trip_error_values.append(trip_error)
        jsd_values.append(density_error)  
        abs_length_error_values.append(absolute_length_error)
        

    # Calculate statistics
    density_error_jsd_avg = np.average(jsd_values)
    density_error_jsd_std = np.std(jsd_values)
    counting_query_error_avg = np.average(counting_query_error_values)
    counting_query_error_std = np.std(counting_query_error_values)
    trip_error_jsd_avg = np.average(trip_error_values)
    trip_error_jsd_std = np.std(trip_error_values)
    abs_length_error_jsd_avg = np.average(abs_length_error_values)
    abs_length_error_jsd_std = np.std(abs_length_error_values)


    #### SAVE RESULTS
    with open(OUTPUT_FILE, mode="w") as file:
        file.write("Metric,Average,Standard Deviation\n")
        file.write(f"Density Error JSD,{density_error_jsd_avg},{density_error_jsd_std}\n")
        file.write(f"Counting Query Error,{counting_query_error_avg},{counting_query_error_std}\n")
        file.write(f"Trip Error JSD,{trip_error_jsd_avg},{trip_error_jsd_std}\n")
        file.write(f"Average Length Error ,{abs_length_error_jsd_avg},{abs_length_error_jsd_std}\n")


    print(f"Results saved in {OUTPUT_FILE}")
    
    

