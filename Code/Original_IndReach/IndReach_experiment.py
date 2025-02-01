#PACKAGES
import ast
import csv
import os
import networkx as nx
import osmnx as ox
import pickle
import random
import itertools
import numpy as np
from collections import Counter


############## Reachability & Markov Sampling ##############################################
def ReacheableSet(G, previous_node,PRECOMPUTED_REACHABLE_SETS):
    if previous_node is None:
        return list(G.nodes())
    else:
        return list(PRECOMPUTED_REACHABLE_SETS.get(previous_node, {}))
        
def MarkovModel(G, node, prev_node,PRECOMPUTED_REACHABLE_SETS): 
    S = ReacheableSet(G, prev_node,PRECOMPUTED_REACHABLE_SETS)
    return 1 / len(S) if node in S else 0

def MaximumDistanceInSubset(G, S,PRECOMPUTED_PATHS):
    max_distance = 0
    if len(S) <= 1:
        return 1
    for node1, node2 in itertools.combinations(S, 2):
        distance = PRECOMPUTED_PATHS.get(node1).get(node2)
        max_distance = max(max_distance, distance)
    return max_distance
    
################ Original Exponential mechanism ########################
def exponential_mechanism(current_node, previous_node, G, epsilon, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS):
    
    #Compute the range of possible reachable nodes
    node_list = ReacheableSet(G, previous_node,PRECOMPUTED_REACHABLE_SETS)
    
    #Compute the score for each possible output
    scores = [-PRECOMPUTED_PATHS.get(current_node).get(v) for v in node_list]
    
    #Compute the probability of each possible ouput base on the scores
    probabilities = [np.exp(epsilon * scores[i] / (2 * diameter))  for i in range(len(scores))]
    probabilities = probabilities / np.sum(probabilities)
    
    return random.choices(population=node_list, weights=probabilities, k=1)[0]
############################################################################################


############################### DTW ##########################################################
def calculate_dtw_with_graph(seq1, seq2, PRECOMPUTED_PATHS):
    n, m = len(seq1), len(seq2)
    
    # Compute matrix DTW initialized at infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0  # Punto inicial
    
    # Compute matrix DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            node1 = seq1[i - 1]
            node2 = seq2[j - 1]
            cost = PRECOMPUTED_PATHS.get(node1).get(node2)
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # Insertion
                dtw_matrix[i, j - 1],    # Elimination
                dtw_matrix[i - 1, j - 1] # Substitution
            )
    
    costo_total = dtw_matrix[n, m]
    return costo_total

######### Confidence interval
def calculate_alpha_for_beta(initial_errors, beta=0.10):
    """
    Compute the value of alpha given beta so Pr(Error > alpha) < beta.
    Args:
        initial_errors (list): List of absolute initial errors between original and anonymized nodes.
        beta (float): Probability of error exceeding alpha (complement of confidence level).

    Returns:
        alpha (float): Value of threshold for alpha.
    """
    # Calculate percentile 1-beta (high percentile)
    confidence_level = 1 - beta
    alpha = np.percentile(initial_errors, confidence_level * 100)
    
    # Calculate proportion of errors bigger than alpha (validation)
    proportion_greater_than_alpha = np.mean(np.array(initial_errors) > alpha)
    
    print(f"Alpha (Pr(Error > alpha) < {beta}): {alpha:.4f}")
    print(f"Proportion of errors > alpha: {proportion_greater_than_alpha:.4f}")
    return alpha

#################################  MAIN ######################################

# ADAPTATIVE INDREACH (Cormode)
# Program to anonymize a whole dataset
def composed_EM(trajectories,max_len, eps, G, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval):

    #GLOBAL PARAMETERS
    # Spatial protection parameters
    round_eps = eps / max_len  # event-level epsilon
    
    print(f"Spatial protection parameters: eps={round_eps}")
    
    # List for output trajectories
    anon_trajectories = []

    # List for local metrics 
    dtw = [] 
    norm_dtw = []
    initial_dist = []
    avg_node_dist = []
    #Specific to length protection:
    length_error = []
    start_time_shift = []
    end_time_shift = []
    
    #START ANONIMIZATION 
    
    for entry in trajectories:
    
        previous_node = None # New trajectory implies we do not have reachability constraints yet
        
        # TIME PROCESSING

        timestamp = entry[0]  # Save the original time in UNIX format
        # Add the time perturbation r*interval seconds
        time_anon = timestamp #There is not time-shift so we save the original timestamp
        start_time_shift.append(0)
        end_time_shift.append(0)
        
        # LOCATIONS PROCESSING

        traj = entry[1]  # Save the original trajectory

        traj_anon = []  # Empty list for the output trajectories

        n = len(traj)  # Original length
        
        dist = []

        # Exponential mechanism for each node

        for i in range(0, n):  # For each node we apply the exponential mechanism
                # Node perturbation
                node = traj[i]  # Real node at position i
                node_anon = exponential_mechanism(node, previous_node, G, round_eps, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS) # Perturbed node using exponential mechanism

                # Local Metrics
                if previous_node == None :
                   #Save the error in the first node
                   initial_dist.append(PRECOMPUTED_PATHS.get(node).get(node_anon))
                else: 
                   #Save the error in the current node 
                   dist.append(PRECOMPUTED_PATHS.get(node).get(node_anon))
                
                traj_anon.append(node_anon)  # Add the new node to the perturbed trajectory

                previous_node = node  # Set the memory for the Reachability model
        
        # Save the new anonymized trajectory
        anon_trajectories.append((time_anon, traj_anon))
        
        #Local metrics per trajectory:
        #1. Intial error was already computed
        #2. After perturbing all nodes we save the average error per node without taking the intial node into account    
        avg_node_dist.append(np.average(dist))
        #3.DTW
        d = calculate_dtw_with_graph(traj, traj_anon, PRECOMPUTED_PATHS) #Absolute
        dn = d/len(traj) #Normalized
        dtw.append(d)
        norm_dtw.append(dn)
        # Length error
        l = len(traj_anon)
        length_error.append(abs(n - l))

    #After perturbing all trajectories we need to save the local metrics
    avg_distance = [
        np.average(start_time_shift),
        np.average(end_time_shift),
        np.average(dtw),
        np.average(norm_dtw),
        np.average(initial_dist),
        np.average(avg_node_dist),
        np.average(length_error),
    ]
    std_distance = [
        np.std(start_time_shift),
        np.std(end_time_shift),
        np.std(dtw),
        np.std(norm_dtw),
        np.std(initial_dist),
        np.std(avg_node_dist),
        np.std(length_error),
    ]
            
    print(f"Anonymization process complete. Total trajectories anonymized: {len(anon_trajectories)}")
    return anon_trajectories, avg_distance, std_distance

#ITERATION
    
def iteration(mapped_trajectories, max_len, eps_s, G, diameter, data_type, i, N,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval):

    anon_trajectories, avg_distance, std_distance = composed_EM(mapped_trajectories, max_len, eps_s, G, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval)
    
    # Calculate alpha-beta accuracy empirically (no normal distribution)
    alpha = calculate_alpha_for_beta(initial_errors=avg_distance[4:],beta=0.10) # Confidence (90%)
    
    # Create directory of results if it does not exist
    output_path1 = f"./Results_{data_type}_Original_epsS{eps_s}_N{N}_{max_len}/Trajectories"
    output_path2 = f"./Results_{data_type}_Original_epsS{eps_s}_N{N}_{max_len}"
    if not os.path.exists(output_path1):
        os.makedirs(output_path1, exist_ok=True)
    if not os.path.exists(output_path2):
        os.makedirs(output_path2, exist_ok=True)
        
    # Saved anonimized trajectories
    with open(f"{output_path1}/anon{i}.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(anon_trajectories)
    
    # Save lengths with headers
    with open(f"{output_path2}/distances{i}.csv", "w", newline='') as f:
        wr = csv.writer(f)
        # Write headers
        # 
        wr.writerow(["start_time_shift", "end_time_shift", "dtw", "normalized_dtw", "initial_error", "sum_node_dist", "length_error", "alpha"])
        # Write average data
        wr.writerow(avg_distance + [alpha])
        # Write standard deviations
        wr.writerow(std_distance + ["-"])

    return avg_distance,alpha
        
########################## function to iterate ###########################################
# Experiment executes INDREACH 10 times on the same original database and saves the results
def Experiment_EM_n(eps_s, max_len, N, data_type, original_file,time_interval):
    # UPLOAD THE ROAD NETWORK
    input_file = f"{data_type}_Databases/Pre_processing_Graph_Extraction/saved_road_data.pkl"
    
    #Warning
    if not os.path.exists(input_file):
         raise FileNotFoundError(f"{input_file} does not exits.")

    # Upload Graph data
    with open(input_file, "rb") as file:
        data = pickle.load(file)
    G = data["G"]
    diameter = data["diameter"]
    PRECOMPUTED_PATHS = data["PRECOMPUTED_PATHS"]
    PRECOMPUTED_REACHABLE_SETS = data["PRECOMPUTED_REACHABLE_SETS"]

    print("Graph Data Correctly Imported:")

    #Total spatil epsilon
    eps = eps_s
    print("Total Spatial Epsilon=",eps)
    
    #open and process the original database
    with open(original_file, "r") as infile:
        reader = csv.reader(infile)
        sampled_trajectories = [
            (int(row[0]), ast.literal_eval(row[1]))  # Lee las trayectorias
            for row in reader
        ]

    # Create the directory of results if it does not exist
    output_path = f"./Results_{data_type}_Original_epsS{eps_s}_N{N}_{max_len}"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Save original data in a single directory
    with open(f"{output_path}/original_data.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(sampled_trajectories)
    
    #Create empty lists for utility metrics
    dtw = []
    norm_dtw = []
    avg_node_dist = []
    initial_error = []
    initial_time_shift = []  
    final_time_shift = []
    length = []
    initial_alpha = []

    #run LenPro 10 times and save the data
    for i in range(10):
        print(f"Experiment round {i}\n")
        distances,alpha=iteration(sampled_trajectories, max_len, eps_s, G, diameter, data_type, i, N,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval)
        
        initial_time_shift.append(distances[0])
        final_time_shift.append(distances[1])
        dtw.append(distances[2])
        norm_dtw.append(distances[3])
        initial_error.append(distances[4])
        avg_node_dist.append(distances[5])
        length.append(distances[6])
        initial_alpha.append(alpha)
    
    avg_distance=[np.average(initial_time_shift),np.average(final_time_shift),np.average(dtw),np.average(norm_dtw),np.average(initial_error),np.average(avg_node_dist),np.average(length)]
    std_distance=[np.std(initial_time_shift),np.std(final_time_shift),np.std(dtw),np.std(norm_dtw),np.std(initial_error),np.std(avg_node_dist),np.std(length)]
    avg_alpha=np.average(initial_alpha)
    std_alpha=np.std(initial_alpha)
     # Save lengths with headers
    with open(f"{output_path}/Summary_Distances.csv", "w", newline='') as f:
        wr = csv.writer(f)
        # Write headers
        wr.writerow(["start_time_shift", "end_time_shift", "dtw", "normalized_dtw", "initial_error", "sum_node_dist", "length_error", "alpha"])
        # Write average data
        wr.writerow(avg_distance + [avg_alpha])
        # Write standard deviations
        wr.writerow(std_distance + [std_alpha])
    
################################################################################



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
