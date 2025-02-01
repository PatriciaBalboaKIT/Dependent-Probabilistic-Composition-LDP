import osmnx as ox
import networkx as nx
import csv

# read CSV
input_file = 'train.csv'
sample_db_file = 'new_graph_mapped_trajectories.csv'

# default centrepoint if no other value is given
PORTO_CENTREPOINT = (41.1474557, -8.5870079)
RADIUS = 2688


def map_trajectories():
    shorter_trajectories = []

    # Create a smaller subgraph
    print("Extracting a smaller subgraph")
    G = ox.graph_from_point(PORTO_CENTREPOINT, RADIUS, dist_type='bbox', network_type='drive', simplify=True)
    if not nx.is_strongly_connected(G):
        G = nx.subgraph(G, max(nx.strongly_connected_components(G), key=len))

    # Define the subgraph's bounding box (north, south, east, west)
    subgraph_bbox = ox.utils_geo.bbox_from_point(PORTO_CENTREPOINT, RADIUS, project_utm=False)
    north, south, east, west = subgraph_bbox

    # Get raw data
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        next(reader)

        for row in reader:
            timestamp = int(row['TIMESTAMP'])
            coordinate_string = row['POLYLINE']
            points = coordinate_string.replace('[', '').replace(']', '').split(',')
            if len(points) < 2:
                continue
            float_list = [float(coord) for coord in points]
            coordinate_list = [(float_list[i], float_list[i + 1]) for i in range(0, len(float_list), 2)]

            # Check if the majority of points are within the subgraph's bounding box
            points_in_subgraph = [
                (x, y) for x, y in coordinate_list if west <= x <= east and south <= y <= north
            ]
            if len(points_in_subgraph) >= len(coordinate_list):  # e.g., 80% of points must be in the subgraph
                shorter_trajectories.append((timestamp,points_in_subgraph))
            
    # Map the filtered shorter trajectories
    print("Mapping trajectories")
    mapped_trajectories = []
    for timestamp, t in shorter_trajectories:
        x = [coord[0] for coord in t]
        y = [coord[1] for coord in t]
        mapped = ox.distance.nearest_nodes(G, X=x, Y=y, return_dist=False)
        mapped_trajectories.append((timestamp, mapped))

    # Write mapped trajectories
    print("Writing trajectories")
    with open(sample_db_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(mapped_trajectories)



if __name__ == "__main__":
    map_trajectories()
