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

import mpmath
import sympy as sp
from datetime import datetime, timedelta # for time

#REQUIRED FUNCTIONS
from event_eps import bisection_method_modified 




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
    #CHECK
    print("epsilon used this round", epsilon)
    
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
    Calcula la Dynamic Time Warping (DTW) entre dos secuencias basándose en distancias de ruta más corta.
    
    Parámetros:
        seq1: lista de tuplas (nodo, tiempo) de la primera secuencia.
        seq2: lista de tuplas (nodo, tiempo) de la segunda secuencia.
        graph: grafo de NetworkX que representa las conexiones entre nodos.
        
    Retorna:
        costo_total: el costo acumulado de DTW.
        dtw_matrix: matriz de costos acumulados.
    """
    n, m = len(seq1), len(seq2)
    
    # Crear matriz DTW inicializada en infinito
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0  # Punto inicial
    
    # Calcular la matriz DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            node1 = seq1[i - 1]
            node2 = seq2[j - 1]
            cost = PRECOMPUTED_PATHS.get(node1).get(node2)
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # Inserción
                dtw_matrix[i, j - 1],    # Eliminación
                dtw_matrix[i - 1, j - 1] # Sustitución
            )
    
    costo_total = dtw_matrix[n, m]
    return costo_total

######### Confidence interval
def calculate_alpha_for_beta(initial_errors, beta=0.10):
    """
    Calcula el valor de alpha dado beta, tal que Pr(Error > alpha) < beta.
    Args:
        initial_errors (list): Lista de errores iniciales absolutos entre nodos originales y anonimizados.
        beta (float): Probabilidad de que el error exceda alpha (complemento del nivel de confianza).

    Returns:
        alpha (float): Valor del umbral alpha.
    """
    # Calcular el percentil 1-beta (percentil alto)
    confidence_level = 1 - beta
    alpha = np.percentile(initial_errors, confidence_level * 100)
    
    # Calcular la proporción de errores mayores que alpha (validación)
    proportion_greater_than_alpha = np.mean(np.array(initial_errors) > alpha)
    
    print(f"Alpha (Pr(Error > alpha) < {beta}): {alpha:.4f}")
    print(f"Proportion of errors > alpha: {proportion_greater_than_alpha:.4f}")
    return alpha

########## PROBABILISTIC COMPOSITION EPSILON calculation ############################################

##Symbolic variables
eps, n, eps_l, Delta = sp.symbols('eps n eps_l Delta')

## Define the formula
formula =2*sp.ln (
    sp.exp(-2 * n * eps_l / Delta) * (
        1 + (
            1 - sp.exp(-eps_l / Delta)
        ) * (
            sp.exp(eps / 2) * sp.exp(2 * eps_l / Delta) * (
                1 - sp.exp(eps * n / 2) * sp.exp(2 * n * eps_l / Delta)
            ) / (
                1 - sp.exp(eps / 2) * sp.exp(2 * eps_l / Delta)
            )
        )
    )
)

# Function to compute f(eps) for given parameters
def calcular_f(n_value, eps_l_value, s_value):
    """
    Substitute specific values into the formula and create a numerical function.
    """
    esp_con_valores = formula.subs({n: n_value, eps_l: eps_l_value, Delta: s_value})
    esp_simplificado = sp.simplify(esp_con_valores)
    esp_func = sp.lambdify(eps, esp_simplificado, modules=["numpy"])
    return esp_func



#################################  MAIN ######################################

## LenPro algorithm (Ours) 

def probabilistically_composed_EM(mapped_trajectories, eps_s, eps_l, len_sensitivity, max_len, G, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval):
    # GLOBAL PARAMETERS
    # Length protection parameters:
    p = 1 - sp.exp(-eps_l / len_sensitivity)

    # Spatial protection parameters
    y = eps_s
    l = y / max_len # Max length allowed in the trajectory database is max_len
    u = l+1
    tolerance = 1e-15
    function_f = calcular_f(max_len, eps_l, len_sensitivity)
    # Run bisection method
    round_eps = bisection_method_modified(function_f, y, l, u, tolerance) #event-level epsilon
    
    print(f"Spatial protection parameters: eps={eps}, l={l}, p={p}")

    # General lists
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

    # START ANONYMIZATION

    for entry in mapped_trajectories:
    
        previous_node = None  # New trajectory implies we do not have reachability constraints yet

        # Compute length protection parameters
        a = np.random.geometric(p) - 1
        r = np.random.geometric(p) - 1
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
        m = n - r + a  # Output length
        
        # OPTION A: Trajectory removed
        if m <= 0:  
            print(f"m={m}, No trajectory output.")
            continue
            
        # OPTION B: Exponential mechanism
        elif m > 0 and (n - r) > 0:  # Exponential mechanism to obtain a trajectory
            print(f"use n-r={n-r} locations")
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
                
            #Check Point   
            print("sum of errors",sum(dist))
            print("normalized sum", sum(dist)/(n-r))     
            avg_node_dist.append(sum(dist)/(n-r))
            
            print(f"Extended trajectory randomly with {a} nodes.")
            for j in range(a):  # Extend the trajectory randomly following reachability MM
                node_anon = random.choice(list(ReacheableSet(G, previous_node,PRECOMPUTED_REACHABLE_SETS)))
                traj_anon.append(node_anon)
                previous_node = node_anon
                
        # OPTION C: Random trajectory
        elif m > 0 and (n - r) <= 0:  # Completely random trajectory
            print(f"Generating completely random trajectory. n={n}, r={r}, a={a}")
            for j in range(m):
                node_anon = random.choice(list(ReacheableSet(G, previous_node,PRECOMPUTED_REACHABLE_SETS)))  # The rest of the trajectory is sampled satisfying reachability
                traj_anon.append(node_anon)
                previous_node = node_anon

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

def iteration(mapped_trajectories, max_len, eps_s,eps_l,len_sensitivity, G, diameter, data_type, i, N,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval):

    anon_trajectories, avg_distance, std_distance = probabilistically_composed_EM(mapped_trajectories, eps_s, eps_l, len_sensitivity, max_len, G, diameter,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval)
    
    # Calcular alpha-beta accuracy empíricamente (distribución no normal)
    alpha = calculate_alpha_for_beta(initial_errors=avg_distance[4:],beta=0.10) # Confidence (90%)
    
    # Crear el directorio de resultados si no existe
    # Crear los directorios de resultados si no existen
    output_path1 = f"./Results_{data_type}_MLenPro_epsS{eps_s}_epsL{eps_l}_N{N}_{max_len}/Trajectories"
    output_path2 = f"./Results_{data_type}_MLenPro_epsS{eps_s}_epsL{eps_l}_N{N}_{max_len}"
    if not os.path.exists(output_path1):
        os.makedirs(output_path1, exist_ok=True)
    if not os.path.exists(output_path2):
        os.makedirs(output_path2, exist_ok=True)
        
    # Guardar las trayectorias anónimas
    with open(f"{output_path1}/anon{i}.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(anon_trajectories)
    
    # Guardar las distancias con encabezados
    with open(f"{output_path2}/distances{i}.csv", "w", newline='') as f:
        wr = csv.writer(f)
        # Escribir encabezados
        # Escribir encabezados
        wr.writerow(["start_time_shift", "end_time_shift", "dtw", "normalized_dtw", "initial_error", "sum_node_dist", "length_error", "alpha"])
        # Escribir datos promedio
        wr.writerow(avg_distance + [alpha])
        # Escribir desviaciones estándar
        wr.writerow(std_distance + ["-"])

    return avg_distance,alpha
                

########################## function to iterate ###########################################
# Experiment que ejecuta LenPro 10 veces en un conjunto de datos y guarda los resultados en un solo directorio
def Experiment_EM( eps_s, eps_l, len_sensitivity, max_len, N, data_type, original_file,time_interval):
   
    # UPLOAD THE ROAD NETWORK
    input_file = f"{data_type}_Databases/Pre_processing_Graph_Extraction/saved_road_data.pkl"

    if not os.path.exists(input_file):
         raise FileNotFoundError(f"El archivo {input_file} no existe.")

    # Cargar los datos
    with open(input_file, "rb") as file:
        data = pickle.load(file)

    # Verificar que se cargaron todas las claves necesarias
    required_keys = ["G", "diameter", "PRECOMPUTED_PATHS", "PRECOMPUTED_REACHABLE_SETS"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"El archivo no contiene la clave requerida: {key}")

    # Extraer los objetos
    G = data["G"]
    diameter = data["diameter"]
    PRECOMPUTED_PATHS = data["PRECOMPUTED_PATHS"]
    PRECOMPUTED_REACHABLE_SETS = data["PRECOMPUTED_REACHABLE_SETS"]

    print("Graph Data Correctly Imported:")
    
    # OPEN ORIGINAL TRAJECTORY FILE
    eps=eps_s
    #open and process the original database
    with open(original_file, "r") as infile:
        reader = csv.reader(infile)
        sampled_trajectories = [
            (int(row[0]), ast.literal_eval(row[1]))  # Lee las trayectorias
            for row in reader
        ]

    # Crear el directorio de resultados si no existe
    output_path = f"./Results_{data_type}_MLenPro_epsS{eps_s}_epsL{eps_l}_N{N}_{max_len}"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Guardar los datos originales en el directorio único
    with open(f"{output_path}/original_data.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(sampled_trajectories)
        
    # Create empty lists for  metrics
    dtw = []
    norm_dtw = []
    avg_node_dist = []
    initial_error = []
    initial_time_shift = []  
    final_time_shift = []
    length = []
    initial_alpha = []
    
    # ITERATIONS OF THE LENPRO MECHANISM
    #run LenPro 10 times and save the data
    for i in range(10):
        print(f"Experiment round {i}\n")
        distances,alpha=iteration(sampled_trajectories, max_len, eps_s,eps_l,len_sensitivity, G, diameter, data_type, i, N,PRECOMPUTED_PATHS,PRECOMPUTED_REACHABLE_SETS,time_interval)
        
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
     # Guardar las distancias con encabezados
    with open(f"{output_path}/Summary_Distances.csv", "w", newline='') as f:
        wr = csv.writer(f)
        # Escribir encabezados
        wr.writerow(["start_time_shift", "end_time_shift", "dtw", "normalized_dtw", "initial_error", "sum_node_dist", "length_error", "alpha"])
        # Escribir datos promedio
        wr.writerow(avg_distance + [avg_alpha])
        # Escribir desviaciones estándar
        wr.writerow(std_distance + [std_alpha])
################################################################################
    

    




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
