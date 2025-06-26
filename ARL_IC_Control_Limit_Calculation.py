#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:48:56 2024

@author: tbeene
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import SSMEWMA
import Q_Chart
import Generate_SPC_MEZCAL_Data_V1
import Multivariate_Change_Point_Approach
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from Generate_SPC_MEZCAL_Data_V1 import Load_Data
from Generate_SPC_MEZCAL_Data_V1 import generate_multivariate_data
from Generate_SPC_MEZCAL_Data_V1 import apply_fixed_clustering
from Generate_SPC_MEZCAL_Data_V1 import apply_DBSCAN
import math
import scienceplots
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.linalg import qr, solve_triangular
from scipy.stats import norm, t
from numpy.linalg import inv
from functools import partial
from numpy.random import multivariate_normal

def write_to_csv(data, filepath, filename):
    data.to_csv(filepath + '/' + filename, sep=',', index=True, encoding='utf-8')

def get_Q_results(full_data, filepath, filename_Q, ARL_list_Q, alpha):


    # Apply Q chart
    result_Q_chart = Q_Chart.Calculate_Q_Chart_UCL_LCL(full_data, alpha)

    count = np.inf


    for key,value in result_Q_chart.items():
        for index,state in enumerate(value.T["States"]):
            if state == 'OOC':
                if index < count:
                    count = index 
    # ARL_OOC
    ARL_list_Q.append(count)
    write_to_csv(result_Q_chart.T, filepath, filename_Q)
       
    
    return result_Q_chart
    
   
    
  
def get_SSEWMA_results(cluster_data, full_data, filepath, filename_SSEWMA, ARL_list_SSEWMA, h):

  
    # Apply SSMEWMA
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(cluster_data, h)
    

    count_SSEWMA = np.inf
    
    for key,value in result_SSEWMA.items():
        for index,state in enumerate(value.T["state"]):
            if state == 'OOC':
                if index < count_SSEWMA:
                    count_SSEWMA = index
        
                

    
    

    ARL_list_SSEWMA.append(count_SSEWMA)
    write_to_csv(result_SSEWMA.T, filepath, filename_SSEWMA)
    
    
    
    

    return result_SSEWMA, count_SSEWMA

def get_combined_results(count_combi, count_combi_Q, full_data, filepath, filename_Combi_Q, ARL_list_Combi, sim, clustered_data_DBSCAN, noise_clusters, filename_Combi_SSEWMA, h):
    # Initialize variables
    observations = len(full_data.T)


    
    # Apply Q-chart to noise data and SSEWMA to cluster data
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(clustered_data_DBSCAN, h)

    if noise_clusters[-1].empty == False:
        result_Q_chart = Q_Chart.Calculate_Q_Chart_UCL_LCL(noise_clusters[-1], 1)
    else:
        result_Q_chart = pd.DataFrame()
    
              
 
        
    for key,value in result_SSEWMA.items():
        if value.T["state"][-1] == 'OOC':
            count_combi = observations


    return result_SSEWMA, result_Q_chart, count_combi

def get_CP_results(full_data, filepath, filename_CP, ARL_list_CP):

    process_readings = len(full_data)
    

    result_CP = Multivariate_Change_Point_Approach.Calculate_Multivariate_Change_Point(full_data)
    
    count_CP = np.inf
    
    for ii, val in enumerate(list(result_CP['state'])):
        if val == 'OOC':
            if ii < count_CP:
                count_CP = ii
    
   
    ARL_list_CP.append(count_CP)
    write_to_csv(result_CP.T, filepath, filename_CP)
    
    return result_CP
    

# Simulate_Q defined here
def simulate_Q(alpha):
    run_length = 200
    runs = 100 # Increase for better accuracy
    sim = 2
    DBSCAN_threshold = 0.80

    ARL_list_Q = []

    data_real, cov = Load_Data(32)

    for run in range(runs):
        filepath = r"C:\Users\tbeene\Desktop\Simulation\Full_Simulation\Full_Simulation_IC"
        filename_Q = f"run_Q_{run}_IC.csv"
        filename_Combi_Q = f"run_Combined_{run}_IC_Q.csv"

        # Generate random process reading
        full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(cov, data_real, run_length)

        result_Q_chart = get_Q_results(full_data.copy(), filepath, filename_Q, ARL_list_Q, alpha)

        write_to_csv(result_Q_chart.T, filepath, filename_Combi_Q)

    # Replace infs with nan for mean calculation
    ARL_list_Q = [v if not np.isinf(v) else np.nan for v in ARL_list_Q]
    ARL_Q = np.nanmean(ARL_list_Q)

    print(f"Alpha: {alpha:.6f}, ARL_Q: {ARL_Q:.4f}") # Debug print

    if np.isnan(ARL_Q):
        return 1000 # Penalize if ARL couldn't be computed
    return ARL_Q - 25 # Goal: find alpha such that ARL_Q = 25


# Simulate_Q defined here
def simulate_SSMEWMA(h):
    run_length = 200
    runs = 100 # Increase for better accuracy
    sim = 2
    DBSCAN_threshold = 0.80

    ARL_list_SSEWMA = []

    data_real, cov = Load_Data(32)

    for run in range(runs):
        filepath = r"C:\Users\tbeene\Desktop\Simulation\Full_Simulation\Full_Simulation_IC"
        filename_SSEWMA = f"run_Q_{run}_IC.csv"
        filename_Combi_Q = f"run_Combined_{run}_IC_Q.csv"

        # Generate random process reading
        full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(cov, data_real, run_length)
        cluster_data = apply_fixed_clustering(full_data, sim, data_real)

        result_SSEWMA_chart = get_SSEWMA_results(cluster_data, full_data, filepath, filename_SSEWMA, ARL_list_SSEWMA, h)

       

    # Replace infs with nan for mean calculation
    ARL_list_SSEWMA = [v if not np.isinf(v) else np.nan for v in ARL_list_SSEWMA]
    ARL_SSEWMA = np.nanmean(ARL_list_SSEWMA)

    print(f"Alpha: {h:.6f}, ARL_Q: {ARL_SSEWMA:.4f}") # Debug print

    if np.isnan(ARL_SSEWMA):
        return 1000 # Penalize if ARL couldn't be computed
    return ARL_SSEWMA - 25 # Goal: find alpha such that ARL_Q = 25

# Simulate_Q defined here
def simulate_PSSMEWMA(h):
    run_length = 200
    runs = 100 # Increase for better accuracy
    sim = 2
    DBSCAN_threshold = 0.80

    ARL_list_Combi = []

    data_real, cov = Load_Data(32)

    for run in range(runs):
        filepath = r"C:\Users\tbeene\Desktop\Simulation\Full_Simulation\Full_Simulation_IC"
        filename_SSEWMA = f"run_Q_{run}_IC.csv"
        filename_Combi_SSEWMA = f"run_Combined_{run}_IC_Q.csv"
        filename_Combi_Q = f"run_Combined_{run}_IC_Q.csv"

        # Generate random process reading
        full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(cov, data_real, run_length)
        cluster_data = apply_fixed_clustering(full_data, sim, data_real)
        
        cluster_matrix = np.zeros((97,run_length+1))
        T_list = []

        count_combi = np.inf
        count_combi_Q = np.inf
        
        for current_run_length in range(2,run_length,1):
            if count_combi != np.inf:
                break

            current_data = full_data.copy().iloc[:,:current_run_length+1]
    
          
            clustered_data_DBSCAN, cluster_data_noise, noise_cluster_number, total_cluster, biggest_cluster, cluster_matrix, T_list = apply_DBSCAN(current_data.T, DBSCAN_threshold, full_data,T_list, cluster_matrix)
            result_SSEWMA, result_Q_chart, count_combi = get_combined_results(count_combi, count_combi_Q, current_data, filepath, filename_Combi_Q, ARL_list_Combi, sim, clustered_data_DBSCAN, cluster_data_noise, filename_Combi_SSEWMA, h)
   
           
 
            
        ARL_list_Combi.append(count_combi)

    # Replace infs with nan for mean calculation
    ARL_list_SSEWMA = [v if not np.isinf(v) else np.nan for v in ARL_list_Combi]
    ARL_SSEWMA = np.nanmean(ARL_list_SSEWMA)

    print(f"Alpha: {h:.6f}, ARL_Q: {ARL_SSEWMA:.4f}") # Debug print

    if np.isnan(ARL_SSEWMA):
        return 1000 # Penalize if ARL couldn't be computed
    return ARL_SSEWMA - 25 # Goal: find alpha such that ARL_Q = 25

# Simulate_Q defined here
def simulate_HC(h):
    run_length = 100
    runs = 20 # Increase for better accuracy
    sim = 2
    DBSCAN_threshold = 0.80

    ARL_list_CP = []

    data_real, cov = Load_Data(32)

    for run in range(runs):
        print(run)
        filepath = r"C:\Users\tbeene\Desktop\Simulation\Full_Simulation\Full_Simulation_IC"
        filename_CP = f"run_Q_{run}_IC.csv"
        filename_Combi_Q = f"run_Combined_{run}_IC_Q.csv"

        # Generate random process reading
        full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(cov, data_real, run_length)

        result_CP_chart = get_CP_results(full_data, filepath, filename_CP, ARL_list_CP)
        print(ARL_list_CP)
       

    # Replace infs with nan for mean calculation
    ARL_list_CP = [v if not np.isinf(v) else np.nan for v in ARL_list_CP]
    
    ARL_CP = np.nanmean(ARL_list_CP)

    print(f"Alpha: {h:.6f}, ARL_CP: {ARL_CP:.4f}") # Debug print

    if np.isnan(ARL_CP):
        return 1000 # Penalize if ARL couldn't be computed
    return ARL_CP - 25 # Goal: find alpha such that ARL_Q = 25


if __name__ == "__main__":
    time1 = time.time()

    # Use Brent's method to find the root
    try:
        # alpha_star = opt.brentq(simulate_Q, 0.000001, 0.001, xtol=1e-4)
        # alpha_star = opt.brentq(simulate_SSMEWMA, 10, 14, xtol=1e-4)
        alpha_star = opt.brentq(simulate_HC, 1, 6, xtol=1e-4)
        print(f"Optimal alpha: {alpha_star}")
    except ValueError as e:
        print(f"Error finding root: {e}")

    print("Time taken:", time.time() - time1)

