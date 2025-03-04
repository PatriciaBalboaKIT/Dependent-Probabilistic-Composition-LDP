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

############### HELPERS #################################
## 1. Reachability & Markov Sampling
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
    
################ Exponential mechanism with Adaptative sensitivity ########################
def exponential_mechanism(current_node, previous_node, G, epsilon, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS):

    node_list = ReacheableSet(G, previous_node,PRECOMPUTED_REACHABLE_SETS)
    
    scores = [-PRECOMPUTED_PATHS.get(current_node).get(v) for v in node_list]
    if previous_node is None:
        probabilities = [np.exp(epsilon * scores[i] / (2 * diameter))  for i in range(len(scores))]
        probabilities = probabilities / np.sum(probabilities)
    else:
        S = ReacheableSet(G,previous_node,PRECOMPUTED_REACHABLE_SETS)
        new_diameter = MaximumDistanceInSubset(G, S,PRECOMPUTED_PATHS)
        probabilities = [np.exp(epsilon * scores[i] / (2 * new_diameter)) * MarkovModel(G,node_list[i],previous_node,PRECOMPUTED_REACHABLE_SETS) for i in range(len(scores))]
        probabilities = probabilities / np.sum(probabilities)
    return random.choices(population=node_list, weights=probabilities, k=1)[0]
############################################################################################



############################### DTW ##########################################################
def calculate_dtw_with_graph(seq1, seq2, PRECOMPUTED_PATHS):
    """
    Compute Dynamic Time Warping (DTW) between the two sequences based on the lengths of the shortest route.
    
    Parameterss:
        seq1: list of tuples (node, time) of the first sequence.
        seq2: list of tuples (node, time) of the second sequence.
        graph: graph of NetworkX representing the connections between nodes.
        
    Return:
        costo_total: acumulated cost of DTW.
        dtw_matrix: matrix of acumulated costs.
    """
    n, m = len(seq1), len(seq2)
    
    # Create matrix DTW inicialized at infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0  # Punto inicial
    
    # Compute matrix DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            node1 = seq1[i - 1]
            node2 = seq2[j - 1]
            cost = PRECOMPUTED_PATHS.get(node1).get(node2)
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # Insert
                dtw_matrix[i, j - 1],    # Delete
                dtw_matrix[i - 1, j - 1] # Substitute
            )
    
    costo_total = dtw_matrix[n, m]
    return costo_total
    
######### Confidence interval
def calculate_alpha_for_beta(initial_errors, beta=0.10):
    """
    Compute the value of alpha given beta, so that Pr(Error > alpha) < beta.
    Args:
        initial_errors (list): List of absolute intials errors between original nodes and anonimized
        beta (float): Probability of error exceeding alpha (complement of confidence level).

    Returns:
        alpha (float): Value of threshold alpha.
    """
    # Compute percentile 1-beta (high percentile)
    confidence_level = 1 - beta
    alpha = np.percentile(initial_errors, confidence_level * 100)
    
    # Compute the proportion of errors bigger than alpha (validationn)
    proportion_greater_than_alpha = np.mean(np.array(initial_errors) > alpha)
    
    print(f"Alpha (Pr(Error > alpha) < {beta}): {alpha:.4f}")
    print(f"Proportion of errors > alpha: {proportion_greater_than_alpha:.4f}")
    return alpha




#################################  MAIN ######################################

# ADAPTATIVE INDREACH (Cormode)

def composed_EM(trajectories,max_len, eps, G, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval):  
    
    #GLOBAL PARAMETERS
    # Spatial protection parameters
    round_eps = eps / max_len  # event-level epsilon
    
    print(f"Spatial protection parameters: eps={round_eps}")
    
    #General lists
    anon_trajectories = []
    #node_dist = []
    dtw = []
    norm_dtw = []
    avg_node_dist = []
    initial_dist = []
    
    # Specific to length protection
    length_error = []
    start_time_shift = []
    end_time_shift = []
    
    #START ANONIMIZATION 
    
    for entry in trajectories:
    
        previous_node = None # New trajectory implies we do not have reachability constraints yet
        
        # Compute length protection parameters
        a = 0
        r = 0
        print(f"Length protection parameters: a={a}, r={r}")
        
        # TIME PROCESSING
        timestamp = entry[0]  # Save the original time in UNIX format
        # Add the time perturbation r*15 seconds
        time_anon = timestamp + r * time_interval
        start_time_shift.append(r * time_interval)
        end_time_shift.append(a * time_interval)
        
        # LOCATIONS PROCESSING

        traj = entry[1]  # Save the original trajectory

        traj_anon = []  # Empty list for the output trajectories

        n = len(traj)  # Original length
        #m = n - r + a  # Output length
        
       
        # OPTION B: Exponential mechanism
        dist = []
        for i in range(r, n):  # For each node we apply the exponential mechanism
                # Node perturbation
                node = traj[i]  # Real node at position i
                print("source node", node)
                
                node_anon = exponential_mechanism(node, previous_node, G, round_eps, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS)  # Perturbed node using exponential mechanism
                print("output node", node_anon)
                
                #Check
                if previous_node == None :
                   print("initial error",PRECOMPUTED_PATHS.get(node).get(node_anon))
                   initial_dist.append(PRECOMPUTED_PATHS.get(node).get(node_anon))
                else: 
                   print("error in the node perturbation",PRECOMPUTED_PATHS.get(node).get(node_anon))
                   dist.append(PRECOMPUTED_PATHS.get(node).get(node_anon))
                
                traj_anon.append(node_anon)  # Add the new node to the perturbed trajectory

                previous_node = node  # Set the memory for the Reachability model
        
        print("sum of errors",sum(dist))
        print("normalized sum", sum(dist)/(n-r))        
        avg_node_dist.append(sum(dist)/(n-r))

        # Save the new anonymized trajectory
        anon_trajectories.append((time_anon, traj_anon))

        # UTILITY METRICS PER TRAJECTORY
        # Spatial errors: Calculate DTW
        d = calculate_dtw_with_graph(traj, traj_anon, PRECOMPUTED_PATHS)
        dn=d/len(traj)
        dtw.append(d)
        norm_dtw.append(dn)
        
        
        
        # Length error
        l = len(traj_anon)
        length_error.append(abs(n - l))


    # UTILITY METRICS OUTSIDE OF THE LOOP (GLOBAL)

    # Counts how many times we make a concrete distance error in the whole anonymization
    #frecuencias = Counter(node_dist)
    
    # Calculate the probability of each error
    #total = len(node_dist)
    #print(total)
    #print(sum(node_dist))

    #prob_errors = {node_dist: count / total for node_dist, count in frecuencias.items()}


    #esperanza_error = sum(node * prob for node, prob in prob_errors.items())
    #print(esperanza_error)

    
    # Save utility metrics
    #Check
    print("avg node error database", np.average(avg_node_dist))

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
    output_path1 = f"./Results_{data_type}_Adapt_epsS{eps_s}_N{N}_{max_len}/Trajectories"
    output_path2 =f"./Results_{data_type}_Adapt_epsS{eps_s}_N{N}_{max_len}"
    if not os.path.exists(output_path1):
        os.makedirs(output_path1, exist_ok=True)
    if not os.path.exists(output_path2):
        os.makedirs(output_path2, exist_ok=True)
        
    # Save the anonimyzed trajectories
    with open(f"{output_path1}/anon{i}.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(anon_trajectories)
    
    # Save the lengths with headers
    with open(f"{output_path2}/distances{i}.csv", "w", newline='') as f:
        wr = csv.writer(f)
        # Write headers
        # 
        wr.writerow(["start_time_shift", "end_time_shift", "dtw", "normalized_dtw", "initial_error", "sum_node_dist", "length_error", "alpha"])
        # Write average data
        wr.writerow(avg_distance + [alpha])
        # Write standar deviations
        wr.writerow(std_distance + ["-"])

    return avg_distance,alpha
        
########################## function to iterate ###########################################
# Experiment executing LenPro 10 times in a set of data and save the results in a single directory
def Experiment_EM_n( eps_s, max_len, N, data_type, original_file,time_interval):
    # UPLOAD THE ROAD NETWORK
    input_file = f"{data_type}_Databases/Pre_processing_Graph_Extraction/saved_road_data.pkl"

    if not os.path.exists(input_file):
         raise FileNotFoundError(f"The file {input_file} does not exist.")

    # Load data
    with open(input_file, "rb") as file:
        data = pickle.load(file)

    # Verify all necesary keys have been loaded
    required_keys = ["G", "diameter", "PRECOMPUTED_PATHS", "PRECOMPUTED_REACHABLE_SETS"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"The file does not contain the required key: {key}")

    # Extract the objects
    G = data["G"]
    diameter = data["diameter"]
    PRECOMPUTED_PATHS = data["PRECOMPUTED_PATHS"]
    PRECOMPUTED_REACHABLE_SETS = data["PRECOMPUTED_REACHABLE_SETS"]

    print("Graph Data Correctly Imported:")
    
    eps = eps_s
    #open and process the original database
    with open(original_file, "r") as infile:
        reader = csv.reader(infile)
        sampled_trajectories = [
            (int(row[0]), ast.literal_eval(row[1]))  # Read trajectories
            for row in reader
        ]

    # Create directory of results if it does not exist
    output_path = f"./Results_{data_type}_Adapt_epsS{eps_s}_N{N}_{max_len}"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Save original data in a unique directory
    with open(f"{output_path}/original_data.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(sampled_trajectories)
    
    #Create empty lists for utility metrics
    initial_time_shift = []  
    final_time_shift = []
    dtw = []
    norm_dtw = []
    initial_error = []
    intial_alpha = []
    avg_node_dist = []
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
     # Save lentghs with headers
    with open(f"{output_path}/Summary_Distances.csv", "w", newline='') as f:
        wr = csv.writer(f)
        # Write headers
        wr.writerow(["start_time_shift", "end_time_shift", "dtw", "normalized_dtw", "initial_error", "sum_node_dist", "length_error", "alpha"])
        # Write average data
        wr.writerow(avg_distance + [avg_alpha])
        # Write standard deviation
        wr.writerow(std_distance + [std_alpha])

    
################################################################################



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
