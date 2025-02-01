import os
import csv
import ast
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def adjust_to_fixed_date(timestamp):
    """
    Adjuts a timestamp so the date is always 1t February 2014, preserving only the original hour and minute
    """
    original_time = datetime.fromtimestamp(timestamp)
    fixed_date = datetime(2014, 2, 1, original_time.hour, original_time.minute, original_time.second)
    return int(fixed_date.timestamp())

def select_trajectories_exponential(
    trajectory_file: str,
    min_length: int,
    max_length: int,
    N: int,
    lambda_: float,
    output_file: str
) -> None:
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"The file {trajectory_file} does not exist.")

    # Read trajectories in the file
    with open(trajectory_file, "r") as infile:
        reader = csv.reader(infile)
        trajectories = [
            (int(row[0]), ast.literal_eval(row[1]))  # Save initial timestamp and trajectory
            for row in reader
        ]
    
    # Filter trajectories by valid lengths and time interval
    filtered_trajectories = [
        (timestamp, trajectory) for timestamp, trajectory in trajectories
        if min_length <= len(trajectory) <= max_length
    ]

    if not filtered_trajectories:
        raise ValueError("There are no trajectories with the required criteria.")

    # Compute probabilities by an inverse exponential distribution
    probabilities = [
        math.exp(-lambda_ * (max_length - length)) for length in range(min_length, max_length + 1)
    ]
    probabilities = [p / sum(probabilities) for p in probabilities]  # Normalizar

    # Compute desired frequencies for each length
    target_counts = {length: round(probabilities[i] * N) for i, length in enumerate(range(min_length, max_length + 1))}

    # Create a dictionary of buckets for length
    length_buckets = {length: [] for length in range(min_length, max_length + 1)}
    for timestamp, trajectory in filtered_trajectories:
        length = len(trajectory)
        length_buckets[length].append((timestamp, trajectory))

    # Select trajectories respecting the desired frequencies
    selected_trajectories = []
    for length, target_count in target_counts.items():
        available = length_buckets[length]
        if len(available) >= target_count:
            selected_trajectories.extend(random.sample(available, target_count))
        else:
            selected_trajectories.extend(available)  # Add all if there are not enough
            print(f"There are not enough trajectories of lenght l={length}, please use a bigger database.")

    # Redistribute missing trajectories
    while len(selected_trajectories) < N:
        remaining_lengths = [
            length for length, available in length_buckets.items()
            if len(available) > 0 and (length, available) not in selected_trajectories
        ]
        if not remaining_lengths:
            break
        for length in remaining_lengths:
            if len(selected_trajectories) >= N:
                break
            selected_trajectories.append(random.choice(length_buckets[length]))

    # Limite number of selected trajectories to N
    selected_trajectories = selected_trajectories[:N]

    # Save selected trajectories in output file with fixed date
    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        for timestamp, trajectory in selected_trajectories:
            adjusted_timestamp = adjust_to_fixed_date(timestamp)
            writer.writerow([adjusted_timestamp, trajectory])

    # Plot distribution of desired lengths
    lengths = [len(trajectory) for _, trajectory in selected_trajectories]
    plt.hist(lengths, bins=range(min_length, max_length + 2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.xticks(range(min_length, max_length + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Usecase example
def main():
    trajectory_file = "new_graph_mapped_trajectories.csv"
    min_length = 2
    max_length = 100
    N = 5000
    output_file = f"Porto_{max_length}_N{N}.csv"
    lambda_ = 0.5  # Parameter for exponential distribution (adjustable)

    try:
        select_trajectories_exponential(trajectory_file, min_length, max_length, N, lambda_, output_file)
        print(f"Selected trajectories saved in {output_file}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()






