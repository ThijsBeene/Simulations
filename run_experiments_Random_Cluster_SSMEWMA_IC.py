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
from Generate_SPC_MEZCAL_Data_V1 import Load_Data, generate_multivariate_data, apply_fixed_clustering, apply_DBSCAN
import Multivariate_Change_Point_Approach
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import scienceplots


def write_to_csv(data, filepath, filename):
    data.to_csv(filepath + '/' + filename, sep=',', index=True, encoding='utf-8')


    
  
def get_SSEWMA_results(cluster_data, full_data, filepath, filename_SSEWMA, ARL_list_SSEWMA, sim):

  
    # Apply SSMEWMA
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(cluster_data, 0)
    
    
    
    
    count_SSEWMA = 0
    count_SSEWMA += np.sum(result_SSEWMA.T['Num_OOC'])
    



    
    ARL_list_SSEWMA.append(count_SSEWMA)
    write_to_csv(result_SSEWMA.T, filepath, filename_SSEWMA)
    
    
    # print(count_SSEWMA)
    # print(result_SSEWMA[val2]['state'])
    # # Check if everything is working correctly!
    # plt.figure()
    # plt.plot(result_SSEWMA[1]['Norm'])
    # plt.plot(result_SSEWMA[1]['Limit'])
    

    return result_SSEWMA, count_SSEWMA


if __name__ == "__main__":
    time1 = time.time()
    cluster_sizes = [2,3,4,6,8,12,16,24]

    # Define global parameters
    run_length = 30
    runs = 500
     
 
    ARL_SSEWMA = []
    
    yerr_list = []

    
    for cluster_size in cluster_sizes:
        
        

        yerrSSEWMA = 0
        ARL_list_SSEWMA = []
        sim = cluster_size
        
        data_real, cov = Load_Data(32)
        
        for run in range(runs):
            filepath = r"C:\Users\tbeene\Desktop\Simulation\Cluster size SSEWMA\Random_Cluster_Assignment_IC"
            filename_SSEWMA = f"run_SSEWMA_{run}_{sim}.csv"
            filename_CP = f"run_CP_{run}_{sim}.csv"
            filename_Q = f"run_Q_{run}_{sim}.csv"
            filename_Combi_Q = f"run_Combined_{run}_{sim}_Q.csv"
            filename_Combi_SSEWMA = f"run_Combined_{run}_{sim}_SSEWMA.csv"
            filename_results = f"run_results_{sim}.csv"
            print(f"Simulation percentage complete: ", 100*(run/runs))
            print(f"cluster size: {cluster_size}")
            
            
            full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(cov, data_real, run_length)
            
            # Generate a random process reading to be OOC
            val = random.randint(1, len(full_data) - 1)
            res = list(full_data.T.keys())[val]
            
            # Calculate random clusters based on fixed cluster size
            cluster_data_SSEWMA = apply_fixed_clustering(full_data, cluster_size, data_real)
            

            result_SSEWMA, count_SSEWMA = get_SSEWMA_results(cluster_data_SSEWMA, full_data, filepath, filename_SSEWMA, ARL_list_SSEWMA, sim)
            print(f"Count: {count_SSEWMA}")
            print(f"Cluster number: {len(cluster_data_SSEWMA)}")
   
        
        
        process_readings = len(full_data)
        ARL_SSEWMA.append((run_length*len(ARL_list_SSEWMA))/np.sum(ARL_list_SSEWMA))
        
        # 95% confidence interval
        yerrSSEWMA = 1.96*(runs*run_length)/((np.sum(ARL_list_SSEWMA)*np.sqrt(np.sum(ARL_list_SSEWMA))))#/(np.sqrt(len(POD_list_combi))) # Standard error of the mean SEM
    
        yerr_list.append(yerrSSEWMA)
        
        


        
        
             
results_overal_df = pd.DataFrame()

process_readings = len(full_data)
  

  


   

results_overal_df['ARL_IC_SSEWMA'] =  ARL_SSEWMA

results_overal_df['yerrSSEWMAARL'] = yerrSSEWMA





results_overal_df['Cluster size'] = cluster_size

write_to_csv(results_overal_df, filepath, filename_results)

print(time.time() - time1)

print(ARL_SSEWMA)
print(yerrSSEWMA)


fig, ax1 = plt.subplots(figsize=(8, 8))
plt.style.use('science')
ax1.errorbar(cluster_sizes, ARL_SSEWMA, yerr=yerr_list, capsize=3, fmt="g--o", ecolor = "black", label = 'ARL')

# plt.plot(cluster_size,ARL_SSEWMA)

ax1.axhline(25, linestyle = '--', c = 'orange', label = '$ARL_{IC}$ = 25')
ax1.set_ylim(0, 40)
ax1.set_xlabel('Cluster size (p) [-]')
ax1.set_ylabel("$ARL_{IC}$ [-]")
ax1.grid()
ax1.legend(loc = 'upper left')
