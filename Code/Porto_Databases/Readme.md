
## Database Pre-Processing for Porto
 
 new_graph_mapped_trajectories.csv contains 383619 trajectories in graph format corresponding to the graph of Porto city center with centre_point = (41.1474557, -8.5870079) and
radius = 2688 meters. All trajectories are uniformly recorded each
 15s.

 For our experiments we sample subdabases from new_graph_mapped.csv. Using the following programs:

 ### unif_database_generator.py
  Parameter we need to input manually:
    trajectory_file = "new_graph_mapped_trajectories.csv"
    min_length = 2
    max_length = 60
    N = 5000
    time_interval = (1391230800, 1393650000) # 1 feb 5am to 1 marz 5 am 2014
  It will select N random trajectories from trajectory_file verifying that the length distribution follows an exponential with lambda=0.6 and maximum length max_len where all trajectories verify that the intial timestamps falls in the give time_interval.

  The output is soted in output_file.

  ### fix_date_database_generator.py
  Parameters we need to input manually:
    trajectory_file = "new_graph_mapped_trajectories.csv"
    min_length = 2
    max_length = 40
    N = 5000
    output_file = f"Porto_{max_length}_N{N}.csv"
    lambda_ = 0.5 

  It will select N random trajectories from trajectory_file verifying that the length distribution follows an exponential with lambda=0.6 and maximum length max_len. Instead of saving the original timestamps, it will save the original time but move all to the same date: 01.02.2014.

  The output is stored in output_file.
