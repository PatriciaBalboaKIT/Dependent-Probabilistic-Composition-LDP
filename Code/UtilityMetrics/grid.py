import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Define parameters of the simulation
# PORTO
#centre_point = (41.1474557, -8.5870079)  # Coordinates
#radius = 2688 # Radius in meters
#DRESDEN
# Default centre point and settings
centre_point = (51.14431557518151, 13.758286754917108)  # Center of the area (lat, lon)
radius = 20000  # Radius in meters


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


# Get the bounding box (min/max lat/lon) of the original graph
def get_bounding_box(G):
    nodes = G.nodes(data=True)
    lats = [data['y'] for node, data in nodes]
    lons = [data['x'] for node, data in nodes]

    # Compute the bounding box
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    return west, south, east, north

west, south, east, north = get_bounding_box(G)

# Define the number of cells along each axis
num_cells_x = 10  # Number of cells in the longitude (x) direction
num_cells_y = 10  # Number of cells in the latitude (y) direction

# Calculate the size of each cell
cell_width = (east - west) / num_cells_x
cell_height = (north - south) / num_cells_y

# Plot the original graph (not the line graph)
fig, ax = ox.plot_graph(G, show=False, close=False)

# Overlay a grid
def draw_grid(ax, west, south, east, north, cell_width, cell_height):
    # Create grid points
    x_ticks = np.arange(west, east + cell_width, cell_width)
    y_ticks = np.arange(south, north + cell_height, cell_height)

    # Draw vertical grid lines
    for x in x_ticks:
        ax.plot([x, x], [south, north], color='gray', linestyle='--', linewidth=1)

    # Draw horizontal grid lines
    for y in y_ticks:
        ax.plot([west, east], [y, y], color='gray', linestyle='--', linewidth=1)

# Draw the grid
#draw_grid(ax, west, south, east, north, cell_width, cell_height)
#output_file = "grid_plot.png"
#plt.savefig(output_file, dpi=300, bbox_inches='tight')
#plt.show()

# Function to map coordinates to grid cells
def map_coords_to_grid(x, y, west, south, cell_width, cell_height):
    grid_x = int((x - west) // cell_width)
    grid_y = int((y - south) // cell_height)
    return (grid_x, grid_y)

def node_to_grid(G, node, west, south, cell_width, cell_height):    
    # Get the coordinates of the node
    x, y = G.nodes[node]['x'], G.nodes[node]['y']

    # Map the midpoint to a grid cell
    grid_cell = map_coords_to_grid(x, y, west, south, cell_width, cell_height)
    return grid_cell

