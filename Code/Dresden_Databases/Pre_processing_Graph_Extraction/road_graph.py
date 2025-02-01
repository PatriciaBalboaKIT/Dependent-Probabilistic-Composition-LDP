import osmnx as ox
import networkx as nx
import pickle

# Default centre point and settings
centre_point = (51.14431557518151, 13.758286754917108)  # Center of area (lat, lon)
radius = 20000  # Radius in meters
time_interval = 6  # Time interval in seconds

#### Road Network Extraction
# Connected graph from map
def roadgraph(centre_point, radius): 
    G = ox.graph_from_point(centre_point, radius, dist_type='bbox', network_type='drive', simplify=True)
    if not nx.is_strongly_connected(G):
        G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    return G
# Save speeds
print("Import graph")
G = roadgraph(centre_point, radius)
G = ox.add_edge_speeds(G, fallback=50)  # Uses 50 km/h if there is no maxspeed
G = ox.add_edge_travel_times(G)

### Reachability and distances

#Reachability taking time into account
def precompute_reachable_sets(V):
    reachable_sets = {}
    for source, paths in nx.all_pairs_dijkstra_path_length(V, weight='travel_time'):
        reachable_sets[source] = {
            target for target, travel_time in paths.items() if travel_time <= time_interval
        }
    return reachable_sets

### EXECUTION
# 2. Precompute shortest paths and diameter
print("Calculating shortest paths for all nodes in G and G diameter")
PRECOMPUTED_PATHS = dict(nx.all_pairs_dijkstra_path_length(G))
diameter = nx.diameter(G)

# 3. Compute reachable sets
print("Compute reachable sets")
PRECOMPUTED_REACHABLE_SETS = precompute_reachable_sets(G)

# 4. Save the objects
output_file = "saved_road_data.pkl"
with open(output_file, "wb") as file:
    pickle.dump({
        "G": nx.MultiDiGraph(G),  # Turn G into a serializable format
        "diameter": diameter,
        "PRECOMPUTED_PATHS": PRECOMPUTED_PATHS,
        "PRECOMPUTED_REACHABLE_SETS": PRECOMPUTED_REACHABLE_SETS
    }, file)

print(f"Data saved in {output_file}.")








