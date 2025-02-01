import csv
import pandas as pd
import numpy as np
from ast import literal_eval

# Function to load timestamps from archive
def load_timestamps(csv_file):
   # Load original trajectories
   trajectories = pd.read_csv(csv_file, header=None).values.tolist()
   timestamps = []     
   for trajectory in trajectories:
        init_time = int(trajectory[0])
        coordinates = literal_eval(trajectory[1])
        for i in range(len(coordinates)):
            timestamps.append(init_time + 15 * i)
   return timestamps
   

        

# Function to ccompute the time intervals within one hour
def calculate_time_intervals(timestamps, output_csv):
    print("total timestamps",len(timestamps))
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    
    print(f"Minimum timestamp: {min_timestamp}")
    print(f"Maximum timestamp: {max_timestamp}")
    
    # Create one hour intervals (3600s)
    # Create the intervals in Unix timestamps
    intervals = np.arange(min_timestamp, max_timestamp+3600 , 3600)
    print(intervals)
    
    # Create DataFrame with time intervals
    # Create the DataFrame directly from the intervals Unix
    time_intervals_df = pd.DataFrame({
        'start': intervals[:-1],  # Every value except the last one
        'end': intervals[1:]      # Every value except the first one
        })
    
    # Save the interval in CSV file
    time_intervals_df.to_csv(output_csv, index=False)
    print(f"Time intervals saved in '{output_csv}'")

    return time_intervals_df

# Main
if __name__ == '__main__':
    # Entry archive with the described format
    input_file = 'Dresden_databases/Dresden_5_N5000.csv'  # Change this with your route
    output_file = 'Dresden_time_intervals_5.csv'
    
    # Load timestamps
    timestamps = load_timestamps(input_file)
    
    # Compute time intervals and save them in a CSV
    time_intervals_df = calculate_time_intervals(timestamps, output_file)




