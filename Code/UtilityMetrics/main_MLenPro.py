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
# Route to the directory where the function is
directory_path = "UtilityMetrics"

# Add directory to sys.path
sys.path.append(directory_path)

# Now you can import the function
from MLenPro_utility import *

######################## PARAMETERS#############################################

parser = ArgumentParser()
parser.add_argument("--max-len", type=int)
parser.add_argument("--data-type")
parser.add_argument("--eps_s", type=float)
parser.add_argument("--eps_l", type=float)
parser.add_argument("--N", type=int)

args = parser.parse_args()

data_type = args.data_type
max_len = args.max_len
N = args.N  # Run on N trajectories

eps_s=args.eps_s

eps_l=args.eps_l
len_sensitivity=1

####################### EXECUTION ##############################
print("calculating utility metrics")
MlenPro_utility(eps_s,eps_l, N, max_len, data_type)

