 #PACKAGES
import sys
import os
import ast
import csv
import networkx as nx
import osmnx as ox
import pickle
import random
import itertools
import numpy as np

from argparse import ArgumentParser


#FUNCTIONS
# Route of directory where the function is
directory_path = "IndReach_adapted_sensitivity"

# Add directory to  sys.path
sys.path.append(directory_path)

# Now you can import the function
from Adapt_IndReach_experiment import *


######################## PARAMETERS#############################################

parser = ArgumentParser()
parser.add_argument("--max-len", type=int)
parser.add_argument("--data-type")
parser.add_argument("--eps_s", type=float)
parser.add_argument("--N", type=int)
parser.add_argument("--interval", type=int)

args = parser.parse_args()

data_type = args.data_type
max_len = args.max_len
N = args.N  # Run on N trajectories
original_file = f"{data_type}_Databases/{data_type}_{max_len}_N{N}.csv"

eps_s=args.eps_s

time_interval=args.interval



######################EXECUTION#####################################################

print("Beginning experiments")
Experiment_EM_n(eps_s,max_len,N,data_type,original_file,time_interval)


#####################################################################################

