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

    # Add special cause variation
    full_data.T[res][start_OOC:] += Shift_Size*np.std(full_data.copy().T[res][:start_OOC])
   
    
    # Apply Q chart
    result_Q_chart = Q_Chart.Calculate_Q_Chart_UCL_LCL(full_data, 3.891)

    count = 0


  
    # Count number of observations before OOC
    for jj in range(start_OOC,run_length,1):
        if result_Q_chart[res]["States"][jj] == "OOC" and count == 0:
            count += jj - start_OOC + 1
            break

    # ARL_OOC
    ARL_list_Q.append(count)
    # Store csv with run results
    write_to_csv(result_Q_chart.T, filepath, filename_Q)
       
    
   
  
def get_SSEWMA_results(cluster_data, full_data, filepath, filename_SSEWMA, ARL_list_SSEWMA):

    process_readings = len(full_data)
    
    val2 = None
    # Add special cause variation
    for key, df in cluster_data.items():
        if res in df.T.columns:
            initial_observation_vector_SSEWMA = cluster_data.copy()[key].T[res][:start_OOC]
            cluster_data[key].T[res][start_OOC:] += Shift_Size*np.std(initial_observation_vector_SSEWMA)
            val2 = key
      
           
    # Apply SSMEWMA
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(cluster_data, 0)
    
    
    
    
    count_SSEWMA = 0
   
    
    # Count number of observations before OOC
    for jj in range(start_OOC,run_length-1,1):
        if result_SSEWMA[val2]['state'][jj] == "OOC" and count_SSEWMA == 0:
            count_SSEWMA += jj - start_OOC + 1
            break
           

    ARL_list_SSEWMA.append(count_SSEWMA)
    # Store csv with run results
    write_to_csv(result_SSEWMA.T, filepath, filename_SSEWMA)
    
    

    return result_SSEWMA, count_SSEWMA

def get_combined_results(count_combi, full_data, filepath, filename_Combi_Q, ARL_list_Combi, clustered_data_DBSCAN, noise_clusters, filename_Combi_SSEWMA, res):
    # Initialize variables
    observations = len(full_data.T)
    val2 = None


    # Apply special cause variation
    if observations >= start_OOC:
        if res in list(noise_clusters[-1].T.columns):
            initial_observation_vector_Q = noise_clusters.copy()[-1].T[res][:start_OOC]
            noise_clusters[-1].T[res][start_OOC:] += Shift_Size*np.std(initial_observation_vector_Q)
                
            
        for key, values in clustered_data_DBSCAN.items():
            if res in list(values.T.columns):
                initial_observation_vector = clustered_data_DBSCAN.copy()[key].T[res][:start_OOC]
                clustered_data_DBSCAN[key].T[res][start_OOC:] += Shift_Size*np.std(initial_observation_vector)
                val2 = key
                
    
    # Apply Q-chart to noise data and SSEWMA to cluster data
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(clustered_data_DBSCAN, 1)
    result_Q_chart = Q_Chart.Calculate_Q_Chart_UCL_LCL(noise_clusters[-1], 3.991)
    
    
    
    # Detect special cause variation
    if val2 != None:
        # Count number of observations before OOC
        if result_SSEWMA[val2]['state'][-1] == "OOC" and count_combi == 0:
            count_combi = observations - start_OOC + 1
        

           
    if val2 == None:
        # Count number of observations before OOC
        if result_Q_chart[res]['States'][-1] == "OOC" and count_combi == 0:
            count_combi = observations - start_OOC + 1
            

    

    return result_SSEWMA, result_Q_chart, count_combi

def get_CP_results(full_data, filepath, filename_CP, ARL_list_CP):

    process_readings = len(full_data)
    
    
    # Add special cause variation
    full_observation_vector = full_data.T[res][:start_OOC]
    full_data.T[res][start_OOC:] += Shift_Size*np.std(full_observation_vector)
   
   
    # Apply MCPD
    result_CP = Multivariate_Change_Point_Approach.Calculate_Multivariate_Change_Point(full_data)

    count_CP = 0
    
    

    for jj in range(start_OOC,run_length,1):
        if result_CP['state'][jj] == "OOC" and count_CP == 0:
            count_CP += jj - start_OOC + 1
            break
       

  
    ARL_list_CP.append(count_CP)
    # Store csv with run results
    write_to_csv(result_CP.T, filepath, filename_CP)
    
    return result_CP
    

if __name__ == "__main__":
    
    # Define global parameters
    run_length = 30
    start_OOC = 20
    runs = 1000
    cluster_size = 6
    DBSCAN_threshold = 0.85
    
    time1 = time.time()
    
    POD_combi = []
    POD2_combi = []
    POD5_combi = []
    
    ARL_Combi = []
 
    yerrCombi_list = []
    yerrCombi2_list = []
    yerrCombi5_list = []
  
    
    Shifts = []
    
    
    POD_list_combi = []
    POD2_list_combi = []
    POD5_list_combi = []
    
    POD_list_SSEWMA = []
    POD2_list_SSEWMA = []
    POD5_list_SSEWMA = []
    
    POD_list_Q = []
    POD2_list_Q = []
    POD5_list_Q = []
    
    POD_list_CP = []
    POD2_list_CP = []
    POD5_list_CP = []
    
    for shift_size in np.arange(0, 8, 1):
        Shift_Size = shift_size
        
        
        ARL_list_Q = []
        ARL_list_SSEWMA = []
        ARL_list_Combi = []
        ARL_list_CP = []
   
        
        
        
        tot_noise_cluster_number = []
        tot_total_cluster_list = []
        tot_biggest_cluster_list = []
        
        data_real, cov = Load_Data(32)
        
        for run in range(1, runs+1, 1):
            # Define data storage locations
            filepath = r"C:\Users\tbeene\Desktop\Simulation\Full_Simulation\Full_Simulation_OOC"
            filename_SSEWMA = f"run_SSMEWMA_{run}_{np.round(shift_size,2)}.csv"
            filename_CP = f"run_MCPD_{run}_{np.round(shift_size,2)}.csv"
            filename_Q = f"run_Q_{run}_{np.round(shift_size,2)}.csv"
            filename_Combi_Q = f"run_Combined_{run}_{np.round(shift_size,2)}_Q.csv"
            filename_Combi_SSEWMA = f"run_Combined_{run}_{np.round(shift_size,2)}_SSEWMA.csv"
            filename_results = f"global_results_{np.round(shift_size,2)}.csv"
            print(f"Simulation percentage complete: ", 100*(run/runs))
            print(f"Shift size: {shift_size}")
            
            
         

            full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(cov, data_real, run_length)
            
            # Generate a random process reading to be OOC
            val = random.randint(1, len(full_data) - 1)
            res = list(full_data.T.keys())[val]
            
            # Calculate random clusters based on fixed cluster size
            cluster_data_SSEWMA = apply_fixed_clustering(full_data, cluster_size, data_real)
            
            # Q, SSMEWMA, MCPD results
            results_Q = get_Q_results(full_data.copy(), filepath, filename_Q, ARL_list_Q)
            result_SSEWMA, count_SSEWMA = get_SSEWMA_results(cluster_data_SSEWMA, full_data.copy(), filepath, filename_SSEWMA, ARL_list_SSEWMA)
            results_CP = get_CP_results(full_data.copy(), filepath, filename_CP, ARL_list_CP)
            
            # Get combination chart results by reclustering every timestep
            count_combi = 0
            for current_run_length in range(20,run_length,1):
                # Speed up simulations by stopping when OOC is detected
                if count_combi != 0:
                    break

                # Get df of observations until and including the final observation
                current_data = full_data.copy().iloc[:,:current_run_length+1]
                
                # Recluster with DBSCAN based on current data set
                clustered_data_DBSCAN, cluster_data_noise, noise_cluster_number, total_cluster, biggest_cluster = apply_DBSCAN(current_data.T, DBSCAN_threshold, data_real)
                
                # Determine the SSEWMA and Q results based on the reclustered data
                results_SSEWMA, result_Q_chart, count_combi = get_combined_results(count_combi, current_data, filepath, filename_Combi_Q, ARL_list_Combi, clustered_data_DBSCAN, cluster_data_noise, filename_Combi_SSEWMA, res)
                
        

                tot_noise_cluster_number.append(noise_cluster_number)
                tot_total_cluster_list.append(total_cluster)
                tot_biggest_cluster_list.append(biggest_cluster)
                
  
            # ARL_OOC
            ARL_list_Combi.append(count_combi)
            # Store csv with run results for combined chart
            write_to_csv(result_SSEWMA.T, filepath, filename_Combi_SSEWMA)
            write_to_csv(result_Q_chart.T, filepath, filename_Combi_Q)
          
            
        POD5 = len([x for x in ARL_list_Combi if 0 < x <= 5])
        POD2 = len([x for x in ARL_list_Combi if 0 < x <= 2])
        POD10 = len([x for x in ARL_list_Combi if 0 < x <= 10])
        
        POD_list_combi.append((POD10/len(ARL_list_Combi)))
        POD2_list_combi.append((POD2/len(ARL_list_Combi)))
        POD5_list_combi.append((POD5/len(ARL_list_Combi)))
            
               
                
        # Calculate the POD by counting all ARL within a given bound
        # Index 0 indicates that nothing was detected!
        POD5 = len([x for x in ARL_list_SSEWMA if 0 < x <= 5])
        POD2 = len([x for x in ARL_list_SSEWMA if 0 < x <= 2])
        POD10 = len([x for x in ARL_list_SSEWMA if 0 < x <= 10])
        
        POD_list_SSEWMA.append((POD10/len(ARL_list_SSEWMA)))
        POD2_list_SSEWMA.append((POD2/len(ARL_list_SSEWMA)))
        POD5_list_SSEWMA.append((POD5/len(ARL_list_SSEWMA)))
        
        
        # Q chart
        POD5 = len([x for x in ARL_list_Q if 0 < x <= 5])
        POD2 = len([x for x in ARL_list_Q if 0 < x <= 2])
        POD10 = len([x for x in ARL_list_Q if 0 < x <= 10])
        
        POD_list_Q.append((POD10/len(ARL_list_Q)))
        POD2_list_Q.append((POD2/len(ARL_list_Q)))
        POD5_list_Q.append((POD5/len(ARL_list_Q)))
    
    
        # CP method
        POD5 = len([x for x in ARL_list_CP if 0 < x <= 5])
        POD2 = len([x for x in ARL_list_CP if 0 < x <= 2])
        POD10 = len([x for x in ARL_list_CP if 0 < x <= 10])
        
        POD_list_CP.append((POD10/len(ARL_list_CP)))
        POD2_list_CP.append((POD2/len(ARL_list_CP)))
        POD5_list_CP.append((POD5/len(ARL_list_CP)))
        
    
        Shifts.append(Shift_Size)
        
         
    results_overal_df = pd.DataFrame()
    

    process_readings = len(full_data)

  
    # Combination
    POD_list_Combi = np.array(POD_list_combi)
    POD2_list_Combi = np.array(POD2_list_combi)
    POD5_list_Combi = np.array(POD5_list_combi)
       

    # Standard error
    yerrCombi = np.sqrt((POD_list_Combi*(1-POD_list_Combi))/len(POD_list_Combi))
    yerrCombi2 = np.sqrt((POD2_list_Combi*(1-POD2_list_Combi))/len(POD2_list_Combi))
    yerrCombi5 = np.sqrt((POD5_list_Combi*(1-POD5_list_Combi))/len(POD5_list_Combi))
       

    results_overal_df['POD10_Combi'] =  POD_list_Combi 
    results_overal_df['POD2_Combi'] =  POD2_list_Combi
    results_overal_df['POD5_Combi'] = POD5_list_Combi

    results_overal_df['yerrCombi10'] = yerrCombi
    results_overal_df['yerrCombi2'] = yerrCombi2
    results_overal_df['yerrCombi5'] = yerrCombi5
    
    
    # SSEWMA
    POD_list_SSEWMA = np.array(POD_list_SSEWMA)
    POD2_list_SSEWMA = np.array(POD2_list_SSEWMA)
    POD5_list_SSEWMA = np.array(POD5_list_SSEWMA)
       

    # Standard error
    yerrSSEWMA = np.sqrt((POD_list_SSEWMA*(1-POD_list_SSEWMA))/len(POD_list_SSEWMA))
    yerrSSEWMA2 = np.sqrt((POD2_list_SSEWMA*(1-POD2_list_SSEWMA))/len(POD2_list_SSEWMA))
    yerrSSEWMA5 = np.sqrt((POD5_list_SSEWMA*(1-POD5_list_SSEWMA))/len(POD5_list_SSEWMA))
       

    results_overal_df['POD10_SSEWMA'] =  POD_list_SSEWMA 
    results_overal_df['POD2_SSEWMA'] =  POD2_list_SSEWMA
    results_overal_df['POD5_SSEWMA'] = POD5_list_SSEWMA

    results_overal_df['yerrSSEWMA10'] = yerrSSEWMA
    results_overal_df['yerrSSEWMA2'] = yerrSSEWMA2
    results_overal_df['yerrSSEWMA5'] = yerrSSEWMA5

    # Q
    POD_list_Q = np.array(POD_list_Q)
    POD2_list_Q = np.array(POD2_list_Q)
    POD5_list_Q = np.array(POD5_list_Q)
       

    # Standard error
    yerrQ = np.sqrt((POD_list_Q*(1-POD_list_Q))/len(POD_list_Q))
    yerrQ2 = np.sqrt((POD2_list_Q*(1-POD2_list_Q))/len(POD2_list_Q))
    yerrQ5 = np.sqrt((POD5_list_Q*(1-POD5_list_Q))/len(POD5_list_Q))
       

    results_overal_df['POD10_Q'] =  POD_list_Q
    results_overal_df['POD2_Q'] =  POD2_list_Q
    results_overal_df['POD5_Q'] = POD5_list_Q

    results_overal_df['yerrQ10'] = yerrQ
    results_overal_df['yerrQ2'] = yerrQ2
    results_overal_df['yerrQ5'] = yerrQ5
    
    # CP
    POD_list_CP = np.array(POD_list_CP)
    POD2_list_CP = np.array(POD2_list_CP)
    POD5_list_CP = np.array(POD5_list_CP)
       

    # Standard error
    yerrCP = np.sqrt((POD_list_CP*(1-POD_list_CP))/len(POD_list_CP))
    yerrCP2 = np.sqrt((POD2_list_CP*(1-POD2_list_CP))/len(POD2_list_CP))
    yerrCP5 = np.sqrt((POD5_list_CP*(1-POD5_list_CP))/len(POD5_list_CP))
       

    results_overal_df['POD10_CP'] =  POD_list_CP
    results_overal_df['POD2_CP'] =  POD2_list_CP
    results_overal_df['POD5_CP'] = POD5_list_CP

    results_overal_df['yerrCP10'] = yerrCP
    results_overal_df['yerrCP2'] = yerrCP2
    results_overal_df['yerrCP5'] = yerrCP5
    
    
    # Add column containing shifts
    results_overal_df['Shift size'] = Shifts
        
        
# Store csv with global results 
write_to_csv(results_overal_df, filepath, filename_results)



print(time.time() - time1)



# Plot results
shift_size = np.arange(0,8,1)


fig, ax1 = plt.subplots(figsize=(8, 8))

ax1.errorbar(shift_size, POD_list_combi, yerr=yerrCombi, capsize=3, fmt="g--o", ecolor = "black", label = '$POD_{10}$ combined')
# ax1.errorbar(shift_size, POD5_list_combi, yerr=yerrCombi5, capsize=3, fmt="b--o", ecolor = "black", label = '$POD_{5} combined$')
# ax1.errorbar(shift_size, POD2_list_combi, yerr=yerrCombi2, capsize=3, fmt="r--o", ecolor = "black", label = '$POD_{2} combined$')

ax1.errorbar(shift_size, POD_list_SSEWMA, yerr=yerrSSEWMA, capsize=3, fmt="r--o", ecolor = "black", label = '$POD_{10}$ SSEWMA')
# ax1.errorbar(shift_size, POD5_list_SSEWMA, yerr=yerrSSEWMA5, capsize=3, fmt="b--o", ecolor = "black", label = '$POD_{5}$ SSEWMA')
# ax1.errorbar(shift_size, POD2_list_SSEWMA, yerr=yerrSSEWMA2, capsize=3, fmt="r--o", ecolor = "black", label = '$POD_{2}$ SSEWMA')

ax1.errorbar(shift_size, POD_list_Q, yerr=yerrQ, capsize=3, fmt="y--o", ecolor = "black", label = '$POD_{10}$ Q')
# ax1.errorbar(shift_size, POD5_list_Q, yerr=yerrQ5, capsize=3, fmt="b--o", ecolor = "black", label = '$POD_{5}$ Q')
# ax1.errorbar(shift_size, POD2_list_Q, yerr=yerrQ2, capsize=3, fmt="r--o", ecolor = "black", label = '$POD_{2}$ Q')

ax1.errorbar(shift_size, POD_list_CP, yerr=yerrCP, capsize=3, fmt="b--o", ecolor = "black", label = '$POD_{10}$ CP')
# ax1.errorbar(shift_size, POD5_list_CP, yerr=yerrCP5, capsize=3, fmt="b--o", ecolor = "black", label = '$POD_{5}$ CP')
# ax1.errorbar(shift_size, POD2_list_CP, yerr=yerrCP2, capsize=3, fmt="r--o", ecolor = "black", label = '$POD_{2}$ CP')

ax1.set_xlabel('Shift size [$\sigma$]')
ax1.set_ylabel("Probability of detection (POD) [-]")
ax1.grid()
ax1.legend(loc = 'upper left')



#