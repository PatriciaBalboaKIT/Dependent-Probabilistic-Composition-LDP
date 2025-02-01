import csv
import pandas as pd
import numpy as np
from ast import literal_eval

# Function to load timestamps from file
def load_timestamps(csv_file):
   # Cargar trayectorias originales
   trajectories = pd.read_csv(csv_file, header=None).values.tolist()
   timestamps = []     
   for trajectory in trajectories:
        init_time = int(trajectory[0])
        coordinates = literal_eval(trajectory[1])
        for i in range(len(coordinates)):
            timestamps.append(init_time + 15 * i)
   return timestamps
   

        

# Function to compute the time intervals of an hour
def calculate_time_intervals(timestamps, output_csv):
    print("total timestamps",len(timestamps))
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    
    print(f"Timestamp mínimo: {min_timestamp}")
    print(f"Timestamp máximo: {max_timestamp}")
    
    # Create intervals of one hour (3600s)
    # Create the interval in Unix timestamps
    intervals = np.arange(min_timestamp, max_timestamp+3600 , 3600)
    print(intervals)
    
    # Create DataFrame with time intervals
    # Create DataFrame directly from Unix intervals
    time_intervals_df = pd.DataFrame({
        'start': intervals[:-1],  # Todos los valores excepto el último
        'end': intervals[1:]      # Todos los valores excepto el primero
        })
    
    # Save intervals in CSV
    time_intervals_df.to_csv(output_csv, index=False)
    print(f"Time intervals saved in '{output_csv}'")

    return time_intervals_df

# Main
if __name__ == '__main__':
    # Entry file with prescribed format
    input_file = 'Porto_Databases/Porto_100_N5000.csv'  # Change this for your rout
    output_file = 'Porto_time_intervals_100.csv'
    
    # Load timestamps
    timestamps = load_timestamps(input_file)
    
    # Compute time intervals and save them in CSV
    time_intervals_df = calculate_time_intervals(timestamps, output_file)




