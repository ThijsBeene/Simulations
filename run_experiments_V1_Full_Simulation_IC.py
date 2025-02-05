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

def write_to_csv(data, filepath, filename):
    data.to_csv(filepath + '/' + filename, sep=',', index=True, encoding='utf-8')

def get_Q_results(full_data, filepath, filename_Q, ARL_list_Q):


    # Apply Q chart
    result_Q_chart = Q_Chart.Calculate_Q_Chart_UCL_LCL(full_data, 3.891)

    count = 0


    for key,value in result_Q_chart.items():
        count += value.T["States"].count("OOC")
  


    # ARL_OOC
    ARL_list_Q.append(count)
    write_to_csv(result_Q_chart.T, filepath, filename_Q)
       
    # plt.figure()
    # plt.plot(result_Q_chart[res]['Observation'])
    # plt.plot(result_Q_chart[res]['UCL'])
    # plt.plot(result_Q_chart[res]['LCL'])
    
    return result_Q_chart
    
   
    
  
def get_SSEWMA_results(cluster_data, full_data, filepath, filename_SSEWMA, ARL_list_SSEWMA, sim):

    process_readings = len(full_data)
    
    # keys = list(cluster_data[0].T.keys())
       
    # cluster_data[0].T[keys[0]][start_OOC:] += 5*np.std(cluster_data[0].T[keys[0]])
      
    # Apply SSMEWMA
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(cluster_data, sim)
    
    
    
    
    count_SSEWMA = 0

    print(result_SSEWMA.T['Num_OOC'])
    count_SSEWMA += np.sum(result_SSEWMA.T['Num_OOC'])
    



    
    ARL_list_SSEWMA.append(count_SSEWMA)
    write_to_csv(result_SSEWMA.T, filepath, filename_SSEWMA)
    
    # for i in range(len(result_SSEWMA.T)):
    #     plt.figure()
    #     plt.plot(result_SSEWMA[i]['Norm'])
    #     plt.plot(result_SSEWMA[i]['Limit'])
    #     plt.title(f"{result_SSEWMA[i]['h']}")

    
    
    # print(count_SSEWMA)
    # print(result_SSEWMA[val2]['state'])
    # # Check if everything is working correctly!
    # plt.figure()
    # plt.plot(result_SSEWMA[val2]['Norm'])
    # plt.plot(result_SSEWMA[val2]['Limit'])
    

    return result_SSEWMA, count_SSEWMA

def get_combined_results(count_combi, full_data, filepath, filename_Combi_Q, ARL_list_Combi, sim, clustered_data_DBSCAN, noise_clusters, filename_Combi_SSEWMA, res):
    # Initialize variables
    observations = len(full_data.T)


       
    
    # Apply Q-chart to noise data and SSEWMA to cluster data
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(clustered_data_DBSCAN, sim, 1)
    result_Q_chart = Q_Chart.Calculate_Q_Chart_UCL_LCL(noise_clusters[-1], 3.991)
    
  
    
  
    for key,cluster in result_SSEWMA.items():

        if cluster['state'][-1] == 'OOC':
            count_combi += 1
 
        
    for key, value in result_Q_chart.items():
        
        if value["States"][-1] == "OOC":
            count_combi += 1
            print('Q')
            
    # plt.figure()
    # plt.plot(result_SSEWMA[0]['Norm'])
    # plt.plot(result_SSEWMA[0]['Limit'])
           
           
    # Write results to csv
    write_to_csv(result_SSEWMA.T, filepath, filename_Combi_SSEWMA)
    write_to_csv(result_Q_chart.T, filepath, filename_Combi_Q)

    return result_SSEWMA, result_Q_chart, count_combi

def get_CP_results(full_data, filepath, filename_CP, ARL_list_CP, sim):

    process_readings = len(full_data)
    

    # Apply MCPD
    result_CP = Multivariate_Change_Point_Approach.Calculate_Multivariate_Change_Point(full_data)

    count_CP = 0
    
    
    count_CP += list(result_CP['state']).count('OOC')
    

    ARL_list_CP.append(count_CP)
    write_to_csv(result_CP.T, filepath, filename_CP)
    
    return result_CP
    

if __name__ == "__main__":
    
    time1 = time.time()
    

    run_length = 30
    start_OOC = 20
    runs = 150
    

    ARL_list_Combi = []
    ARL_list_SSEWMA = []
    ARL_list_CP = []
    ARL_list_Q = []
 
    
    
    
    sim = 6
    DBSCAN_threshold = 0.85
    
    tot_noise_cluster_number = []
    tot_total_cluster_list = []
    tot_biggest_cluster_list = []
    
    data_real, cov = Load_Data(32)
    
    for run in range(1, runs+1, 1):
        filepath = r"C:\Users\tbeene\Desktop\Simulation\Full_Simulation\Full_Simulation_IC"
        filename_SSEWMA = f"run_SSEWMA_{run}_IC.csv"
        filename_CP = f"run_CP_{run}_IC.csv"
        filename_Q = f"run_Q_{run}_IC.csv"
        filename_Combi_Q = f"run_Combined_{run}_IC_Q.csv"
        filename_Combi_SSEWMA = f"run_Combined_{run}_IC_SSEWMA.csv"
        filename_results = f"run_results_IC.csv"
        print(f"Simulation percentage complete: ", 100*(run/runs))
  
        
        
     

        # Generate a random process reading for OOC
        full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(cov, data_real, run_length)
        
        count_combi = 0
        cluster_data_SSEWMA = apply_fixed_clustering(full_data, sim, data_real)
      
        
        
        
        
        val = random.randint(1, len(full_data) - 1)
        res = list(full_data.T.keys())[val]
        result_Q_chart = get_Q_results(full_data.copy(), filepath, filename_Q, ARL_list_Q)
        result_SSEWMA, count_SSEWMA = get_SSEWMA_results(cluster_data_SSEWMA.copy(), full_data.copy(), filepath, filename_SSEWMA, ARL_list_SSEWMA, sim)
        # results_CP = get_CP_results(full_data.copy(), filepath, filename_CP, ARL_list_CP, sim)
        # clustered_data_DBSCAN, cluster_data_noise, noise_cluster_number, total_cluster, biggest_cluster = apply_DBSCAN(full_data.T, DBSCAN_threshold, full_data)
        # results_SSEWMA, result_Q_chart, count_combi = get_combined_results(count_combi, full_data, filepath, filename_Combi_Q, ARL_list_Combi, sim, clustered_data_DBSCAN, cluster_data_noise, filename_Combi_SSEWMA, res)
        
           
        
        
        
        
        for current_run_length in range(2,run_length,1):

            print(count_combi)
            full_data1 = full_data.copy().iloc[:,:current_run_length]
    
          
            clustered_data_DBSCAN, cluster_data_noise, noise_cluster_number, total_cluster, biggest_cluster = apply_DBSCAN(full_data1.T, DBSCAN_threshold, full_data)
            results_SSEWMA, result_Q_chart, count_combi = get_combined_results(count_combi, full_data1, filepath, filename_Combi_Q, ARL_list_Combi, sim, clustered_data_DBSCAN, cluster_data_noise, filename_Combi_SSEWMA, res)
            
           
       
            
       
    

            tot_noise_cluster_number.append(noise_cluster_number)
            tot_total_cluster_list.append(total_cluster)
            tot_biggest_cluster_list.append(biggest_cluster)
            
  
        # ARL_OOC
        ARL_list_Combi.append(count_combi)
        
      
    process_readings = len(full_data)
  
    ARL_Combi = (runs*run_length)/np.sum(ARL_list_Combi)
    yerrCombi = (runs*run_length)/((np.sum(ARL_list_Combi)*np.sqrt(np.sum(ARL_list_Combi))))#/(np.sqrt(len(POD_list_combi))) # Standard error of the mean SEM
    
    ARL_SSEWMA = (runs*run_length)/np.sum(ARL_list_SSEWMA)
    yerrSSEWMA = (runs*run_length)/((np.sum(ARL_list_SSEWMA)*np.sqrt(np.sum(ARL_list_SSEWMA))))#/(np.sqrt(len(POD_list_combi))) # Standard error of the mean SEM
    
    ARL_Q = (runs*run_length)/np.sum(ARL_list_Q)
    yerrQ = (runs*run_length)/((np.sum(ARL_list_Q)*np.sqrt(np.sum(ARL_list_Q))))#/(np.sqrt(len(POD_list_combi))) # Standard error of the mean SEM
    
    ARL_CP = (runs*run_length)/np.sum(ARL_list_CP)
    yerrCP = (runs*run_length)/((np.sum(ARL_list_CP)*np.sqrt(np.sum(ARL_list_CP))))#/(np.sqrt(len(POD_list_combi))) # Standard error of the mean SEM
    
           
            
    
    
     
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



print(time.time() - time1)


fig, ax1 = plt.subplots(figsize=(8, 8))

ax1.errorbar(1, ARL_Combi, yerr=yerrCombi, capsize=3, fmt="g--o", ecolor = "black", label = '$ARL_{IC}$ combined chart')
ax1.errorbar(2, ARL_SSEWMA, yerr=yerrSSEWMA, capsize=3, fmt="b--o", ecolor = "black", label = '$ARL_{IC}$ SSEWMA chart')
ax1.errorbar(3, ARL_Q, yerr=yerrQ, capsize=3, fmt="r--o", ecolor = "black", label = '$ARL_{IC}$ Q chart')
ax1.errorbar(4, ARL_CP, yerr=yerrCP, capsize=3, fmt="y--o", ecolor = "black", label = '$ARL_{IC}$ MCPD')



ax1.set_xlabel('Index [-]')
ax1.set_ylabel("$ARL_{IC}$ [-]")
ax1.grid()
ax1.legend(loc = 'upper left')



# ax1.set_xlabel('DBSCAN threshold')
# ax1.set_ylabel("POD after 10 observations", color = 'green')

# ax1.grid()
# ax1.legend(loc = 'upper left')

# ax2.set_ylabel("Number of process readings in cluster", color = 'blue')

# ax2.legend(loc = 'center left')
# fig.suptitle("Combination control chart $ARL_{3}$ for varying DBSCAN threshold")
# fig.autofmt_xdate()

# plt.show()