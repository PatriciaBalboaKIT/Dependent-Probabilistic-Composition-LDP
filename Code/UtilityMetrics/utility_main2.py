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
from grid import node_to_grid



############### HELPERS ######################
# ensure probability sums up to 1
def probability_sanity_check(prob_dict):
    total_prob = sum(prob_dict.values())
    if not (0.9 <= total_prob <= 1.1):  # tolerance
        print(f"Warning: Probabilities sum to {total_prob}, expected close to 1.")

#######################################################################################################3

#### INTERVAL MATCHING FUNCTION ######################
def find_interval_cached(timestamp,intervals):
    match = None
    for interval in intervals: 
        start = interval[0]
        end = interval[1]
        if start <= timestamp < end:
            match=(start, end)
    
    if match is None:
       print(f"timestamp {timestamp} out of interval")
    return match
########################################################

############## Global Metrics ##########################
########################################################################################

### 2. COUNTING QUERY ERROR##############
######################################################################################################

def calculate_counting_query_error_vectorized(trajectories, anon_trajectories,LOCATIONS,time_intervals_dict):
    #We build to dictionaries to map the locations (LOCATIONS) and time intervals (time_intervals_dict) to numeric indices
    #This allows us to enter the matrices in an eficient way.
    location_to_index = {loc: i for i, loc in enumerate(LOCATIONS)}
    interval_to_index = {interval: i for i, interval in enumerate(time_intervals_dict.keys())}
    
    #We initialize te matrices of size (number of locations, number of intervals) filled with zeroes
    #These matrices will be used to count the visits on each combination of location and interval, for both the original and anonymized trajectories
    visit_array = np.zeros((len(LOCATIONS), len(time_intervals_dict)), dtype=int)
    anon_visit_array = np.zeros((len(LOCATIONS), len(time_intervals_dict)), dtype=int)
    
    # the function process_trajectory updates a counting matrix(visit_array or anon_visit_array) based on the visits of a specific trajectory.
    def process_trajectory(trajectory, visit_array):
        init_time = int(trajectory[0])
        coordinates = literal_eval(trajectory[1])
        for i, coordinate in enumerate(coordinates):
            current_time = init_time + i * 15
            time_interval = find_interval_cached(current_time,time_intervals_dict)
            if time_interval is not None:
                loc_idx = location_to_index[coordinate]
                interval_idx = interval_to_index[time_interval]
                visit_array[loc_idx, interval_idx] += 1 #Se incrementa el contador correspondiente en la matriz de visitas.

    # Process original trajectories
    for trajectory in trajectories:
        process_trajectory(trajectory, visit_array)

    # Process anonymized trajectories
    for trajectory in anon_trajectories:
        process_trajectory(trajectory, anon_visit_array)

    # Calculate error
    #Calculate error as the difference between the counting matrices:
    error = np.sum(np.abs(visit_array - anon_visit_array))
    counts = np.sum(visit_array)
    #Normalize the error using the total of original visits:
    return error / (counts if counts > 0 else 0.001 *  len(LOCATIONS))
########################################################################################################

############### 3. TRIP ERROR ############################
def get_bounding_box(G):
    nodes = G.nodes(data=True)
    lats = [data['y'] for node, data in nodes]
    lons = [data['x'] for node, data in nodes]

    # Compute the bounding box
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    return west, south, east, north

def calculate_trip_error(G, trajectories, anon_trajectories):
    if not trajectories or not anon_trajectories:
        raise ValueError("Input trajectory lists cannot be empty.")
    
    # Get the bounding box (min/max lat/lon) of the original graph

    west, south, east, north = get_bounding_box(G)

    # Define the number of cells along each axis
    num_cells_x = 10  # Number of cells in the longitude (x) direction
    num_cells_y = 10  # Number of cells in the latitude (y) direction

    # Calculate the size of each cell
    cell_width = (east - west) / num_cells_x
    cell_height = (north - south) / num_cells_y

    # Initialize the counter
    trip_counts = defaultdict(int)
    anon_trip_counts = defaultdict(int)

    # Count original trajectories
    for trajectory in trajectories:
        coordinates = literal_eval(trajectory[1])
        start, end = coordinates[0], coordinates[-1]
        start_cell = node_to_grid(G, start, west, south, cell_width, cell_height)
        end_cell = node_to_grid(G, end, west, south, cell_width, cell_height)
        trip_counts[(start_cell, end_cell)] += 1




    for trajectory in anon_trajectories:
        coordinates = literal_eval(trajectory[1])
        start, end = coordinates[0], coordinates[-1]
        start_cell =  node_to_grid(G, start, west, south, cell_width, cell_height)
        end_cell = node_to_grid(G, end, west, south, cell_width, cell_height)
        anon_trip_counts[(start_cell, end_cell)] += 1

    # Calculate probabilitiess
    trip_probs = {
        (start, end): count / len(trajectories)
        for (start, end), count in trip_counts.items()
    }
    anon_trip_probs = {
        (start, end): count / len(anon_trajectories)
        for (start, end), count in anon_trip_counts.items()
    }

    # Sanity check de probabilidades
    probability_sanity_check(trip_probs)
    probability_sanity_check(anon_trip_probs)

    # Calculate distributions
    dist_original = []
    dist_anon = []

    #Convert dict to list to run JSD
    for pair in set(trip_probs.keys()) | set(anon_trip_probs.keys()):
        orig=trip_probs.get(pair,0)
        anon=anon_trip_probs.get(pair,0)
        if orig != 0 or anon != 0: # otherwise JSD infinity because 1/2(P+Q)=0
                dist_original.append(orig)
                dist_anon.append(anon)

    # Calculate Jensen-Shannon distance
    return jensenshannon(dist_original, dist_anon)


########################################################
    
############## 4. DENSITY ERROR ######################
# number of visits to a certain location (time-independent)
def get_visit_counts(trajectories,LOCATIONS):
    visit_counts = defaultdict(int)
    all_visits = 0
    probs = {}

    # calculate counts
    for trajectory in trajectories:
        coordinates = literal_eval(trajectory[1])

        for coordinate in coordinates:
            visit_counts[coordinate] += 1
            all_visits += 1

    # calculate probabilities: 
    for location in LOCATIONS:
        try:
            probs[location] = visit_counts[location] / all_visits
        except KeyError:
            probs[location] = 0

    # sanity check
    probability_sanity_check(probs)

    return probs, visit_counts

def calculate_density_error(trajectories, anon_trajectories,LOCATIONS):
    probs,_ = get_visit_counts(trajectories,LOCATIONS)
    probs_anon,_ = get_visit_counts(anon_trajectories,LOCATIONS)

    dist_original = []
    dist_anon = []

    for location in LOCATIONS:
        try:
            if probs[location]!=0 or probs_anon[location]!=0: #Otherwise JSD would be inf
                dist_original.append(probs[location])
                dist_anon.append(probs_anon[location]) 
        except KeyError:
            continue
    
    return jensenshannon(dist_original,dist_anon)
#######################################################

### 5. Avg LENGTH ERROR #################################

def length_error(trajectories, anon_trajectories):
    len_real = []
    len_perturbed = []
    
    
    for trajectory in trajectories:
        real_coordinates = literal_eval(trajectory[1])
        len_real.append(len(real_coordinates))
    for anon_trajectory in anon_trajectories:
        perturbed_coordinates = literal_eval(anon_trajectory[1])
        len_perturbed.append(len(perturbed_coordinates))
        
        
    avg_real=np.average(len_real)
    print(avg_real)
    avg_perturbed=np.average(len_perturbed)
    print(avg_perturbed)
    
    abs_error=abs(avg_real-avg_perturbed)
    
    

    return abs_error
    

    
