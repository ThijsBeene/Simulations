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
from Generate_SPC_MEZCAL_Data_V1 import find_groups
import math
import scienceplots
from matplotlib.patches import Rectangle

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

def get_combined_results(count_combi, count_combi_Q, full_data, filepath, filename_Combi_Q, ARL_list_Combi, sim, clustered_data_DBSCAN, noise_clusters, filename_Combi_SSEWMA, count_Q_com, count_SSEWMA_com):
    # Initialize variables
    observations = len(full_data.T)


    
    # Apply Q-chart to noise data and SSEWMA to cluster data
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(clustered_data_DBSCAN, 1)

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
    

    # Apply HC chart
    result_CP = Multivariate_Change_Point_Approach.Calculate_Multivariate_Change_Point(full_data)
 
    count_CP = np.inf
    
    for ii, val in enumerate(list(result_CP['state'])):
        if val == 'OOC':
            if ii < count_CP:
                count_CP = ii
    
   
    ARL_list_CP.append(count_CP)
    write_to_csv(result_CP.T, filepath, filename_CP)
    
    return result_CP
    

if __name__ == "__main__":
    
    time1 = time.time()
    

    run_length = 200
    runs = 10
    sim = 2
    DBSCAN_threshold = 0.80
    

    ARL_list_Combi = []
    ARL_list_SSEWMA = []
    ARL_list_CP = []
    ARL_list_Q = []
 
    
    
    cluster_matrix = np.zeros((97,run_length+1))
    T_list = []
    
    
    tot_noise_cluster_number = []
    tot_total_cluster_list = []
    tot_biggest_cluster_list = []
    
    data_real, cov = Load_Data(32)
    
    count_Q_com = 0
    count_SSEWMA_com = 0
    
    for run in range(runs):
        filepath = r"C:\Users\tbeene\Desktop\Simulation\Full_Simulation\Full_Simulation_IC"
        filename_SSEWMA = f"run_SSEWMA_{run}_IC.csv"
        filename_CP = f"run_CP_{run}_IC.csv"
        filename_Q = f"run_Q_{run}_IC.csv"
        filename_Combi_Q = f"run_Combined_{run}_IC_Q.csv"
        filename_Combi_SSEWMA = f"run_Combined_{run}_IC_SSEWMA.csv"
        filename_results = f"global_results_IC.csv"
        print(f"Simulation percentage complete: ", 100*(run/runs))
  
        
        
     

        # Generate a random process reading for OOC
        full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(data_real, run_length).T
        cluster_data_SSEWMA = apply_fixed_clustering(full_data, sim, data_real)
      
        
 
        result_Q_chart = get_Q_results(full_data.copy(), filepath, filename_Q, ARL_list_Q, 0.0005378341924398626)
        result_SSEWMA_2_L, count_SSEWMA = get_SSEWMA_results(cluster_data_SSEWMA.copy(), full_data.copy(), filepath, filename_SSEWMA, ARL_list_SSEWMA, 12.576804)
        # Limit HC results to fasten calculations
        results_CP = get_CP_results(full_data.iloc[:,:50].copy(), filepath, filename_CP, ARL_list_CP)
        
           
        
        
        

        count_combi = np.inf
        count_combi_Q = np.inf
        
        for current_run_length in range(2,run_length,1):
            print(current_run_length)
            if count_combi != np.inf:
                break

            current_data = full_data.copy().iloc[:,:current_run_length+1]
    
          
            clustered_data_DBSCAN, cluster_data_noise, cluster_size_counts_dict, T_list = find_groups(current_data.T, data_real, T_list, cluster_matrix)
            result_SSEWMA, result_Q_chart, count_combi = get_combined_results(count_combi, count_combi_Q, current_data, filepath, filename_Combi_Q, ARL_list_Combi, sim, clustered_data_DBSCAN, cluster_data_noise, filename_Combi_SSEWMA, count_Q_com, count_SSEWMA_com)
   
           
            
  
        # # ARL_OOC
        
        # Write results to csv for combination chart
        write_to_csv(result_SSEWMA.T, filepath, filename_Combi_SSEWMA)
        write_to_csv(result_Q_chart.T, filepath, filename_Combi_Q)
        ARL_list_Combi.append(count_combi)
      
    process_readings = len(full_data)
    
    


    def calculate_metrics(ARL_list):
        if len(ARL_list) == 0:
            return 0, 0, 0, 0
        else:
            ARL = np.nanmean(ARL_list)
            yerr = np.nanstd(ARL_list) / np.sqrt(len(ARL_list))
            min_val = np.nanmin(ARL_list)
            max_val = np.nanmax(ARL_list)
            return ARL, yerr, min_val, max_val
    
    ARL_list_Combi = [v if not np.isinf(v) else np.nan for v in ARL_list_Combi]
    ARL_Combi, yerrCombi, min_combi, max_combi = calculate_metrics(ARL_list_Combi)
    
    ARL_list_SSEWMA = [v if not np.isinf(v) else np.nan for v in ARL_list_SSEWMA]
    ARL_SSEWMA, yerrSSEWMA, min_SSEWMA, max_SSEWMA = calculate_metrics(ARL_list_SSEWMA)
    
    ARL_list_Q = [v if not np.isinf(v) else np.nan for v in ARL_list_Q]
    ARL_Q, yerrQ, min_Q, max_Q = calculate_metrics(ARL_list_Q)
    
    ARL_list_CP = [v if not np.isinf(v) else np.nan for v in ARL_list_CP]
    ARL_CP, yerrCP, min_CP, max_CP = calculate_metrics(ARL_list_CP)



     
results_overal_df = pd.DataFrame()

   

process_readings = len(full_data)

  


results_overal_df['ARL_Combi'] =  ARL_Combi
results_overal_df['ARL_SSEWMA'] =  ARL_SSEWMA
results_overal_df['ARL_Q'] = ARL_Q
results_overal_df['ARL_CP'] = ARL_CP

results_overal_df['yerrCombi'] = yerrCombi
results_overal_df['yerrSSEWMA'] = yerrSSEWMA
results_overal_df['yerrQ'] = yerrQ
results_overal_df['yerrCP'] = yerrCP



    
write_to_csv(results_overal_df, filepath, filename_results)


# Replace with your actual values
means = [ARL_Combi, ARL_SSEWMA, ARL_Q, ARL_CP]
errors = [yerrCombi, yerrSSEWMA, yerrQ, yerrCP] # SEMs
mins = [min_combi, min_SSEWMA, min_Q, min_CP]
maxs = [max_combi, max_SSEWMA, max_Q, max_CP]

labels = [r'P-SSMEWMA', r'R-SSMEWMA', r'Q-chart', 'HC-chart']
colors = ['green', 'blue', 'red', 'gold']

x = np.arange(len(means))
box_width = 0.4

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(means)):
    mean = means[i]
    ci = 1.96 * errors[i]
    lower = mean - ci
    upper = mean + ci

    # CI box
    rect = Rectangle((x[i] - box_width / 2, lower),
                     box_width, upper - lower,
                     color=colors[i], alpha=0.4, edgecolor='black')
    ax.add_patch(rect)

    # Horizontal black line for the mean
    ax.plot([x[i] - box_width / 2, x[i] + box_width / 2],
            [mean, mean], color='black', linewidth=2, label='Mean' if i == 0 else "")

    # Min–Max with caps
    err_low = mean - mins[i]
    err_high = maxs[i] - mean
    ax.errorbar(x[i], mean, yerr=[[err_low], [err_high]],
                fmt='none', ecolor='black', elinewidth=1.5, capsize=6,
                label='Min/Max' if i == 0 else "")

# Target ARL reference line
ax.axhline(25, linestyle='--', color='orange', linewidth=1.5, label='Target ARL')

# Aesthetics
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=12)
ax.set_ylabel(r'$\mathit{ARL}_{IC}$', fontsize=13)
ax.set_ylim(0, max(maxs) + 10)
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', labelsize=11)
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()



