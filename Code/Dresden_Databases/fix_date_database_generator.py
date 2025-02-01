import os
import csv
import ast
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def adjust_to_fixed_date(timestamp):
    """
    Adjust timestamp so that the date is always 1st of February of 2014, only preserving the original hours and minutes
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

    # Read trajectories from file
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
        raise ValueError("There are no trajectories achieving the specified criteria.")

    # Compute the probabilities by a inversed exponential distribution
    probabilities = [
        math.exp(-lambda_ * (max_length - length)) for length in range(min_length, max_length + 1)
    ]
    probabilities = [p / sum(probabilities) for p in probabilities]  # Normalize

    # Compute desired frequencies for each length
    target_counts = {length: round(probabilities[i] * N) for i, length in enumerate(range(min_length, max_length + 1))}

    # Create a dictionary of buckets by length
    length_buckets = {length: [] for length in range(min_length, max_length + 1)}
    for timestamp, trajectory in filtered_trajectories:
        length = len(trajectory)
        length_buckets[length].append((timestamp, trajectory))

    # Select trajectories respecting desired frequencies
    selected_trajectories = []
    for length, target_count in target_counts.items():
        available = length_buckets[length]
        if len(available) >= target_count:
            selected_trajectories.extend(random.sample(available, target_count))
        else:
            selected_trajectories.extend(available)  # Add all if the available ones are not enough
            print(f"There are not enough trajectories of length l={length}, please use a bigger database.")

    # Redistribute the missing trajectories
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

    # Limit the number of selected trajectories to N
    selected_trajectories = selected_trajectories[:N]

    # Save the selected trajectories in the output file with fixed date
    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        for timestamp, trajectory in selected_trajectories:
            adjusted_timestamp = adjust_to_fixed_date(timestamp)
            writer.writerow([adjusted_timestamp, trajectory])
    # Plot the distribution of selected lengths
    lengths = [len(trajectory) for _, trajectory in selected_trajectories]
    plt.hist(lengths, bins=range(min_length, max_length + 2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.xticks(range(min(lengths), max(lengths) + 1, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{max_length}length_dist.png')  # Save image as PNG
    plt.show()


# Usecase example
def main():
    trajectory_file = "8k_mapped.csv"
    min_length = 2
    max_length = 10
    N = 5000
    output_file = f"Dresden_{max_length}_N{N}.csv"
    lambda_ = 0.5  # Parameter for the exponential distirbution (adjustable)

    try:
        select_trajectories_exponential(trajectory_file, min_length, max_length, N, lambda_, output_file)
        print(f"Selected trajectories saved in {output_file}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()






