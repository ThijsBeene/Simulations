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
from Generate_SPC_MEZCAL_Data_V1 import Load_Data, generate_multivariate_data, apply_fixed_clustering, find_groups
import math
import scienceplots

random.seed(10)

def write_to_csv(data, filepath, filename):
    data.to_csv(filepath + '/' + filename, sep=',', index=True, encoding='utf-8')

def get_Q_results(full_data, filepath, filename_Q, ARL_list_Q):

    # Add special cause variation
    full_data.T[res][start_OOC:] += Shift_Size*np.std(full_data.copy().T[res][:start_OOC])
   
    
    # Apply Q chart
    result_Q_chart = Q_Chart.Calculate_Q_Chart_UCL_LCL(full_data, 0.0005378341924398626)

    count = 0


  
    # Count number of observations before OOC
    for jj in range(start_OOC,run_length,1):
        if result_Q_chart[res]["States"][jj] == "OOC" and count == 0:
            count += jj - start_OOC + 1
            break

    # ARL_OOC
    ARL_list_Q.append(count)


    plt.figure()
    plt.grid()
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=18)
    plt.rc('figure', titlesize=16)
    plt.plot(np.arange(0,30,1), result_Q_chart[res]['Observation'], label = r"$x_n$")
    plt.plot(np.arange(0,30,1),result_Q_chart[res]['LCL'], c = 'r', label = 'UCL/LCL')
    plt.plot(np.arange(0,30,1),result_Q_chart[res]['UCL'], c = 'r')
    plt.legend()
    plt.ylabel(r"$x_n$ [-]")
    plt.xlabel('n [-]')
       
    
   
  
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
    result_SSEWMA = SSMEWMA.Calculate_SSEWMA_Norm_Lim(cluster_data, 12.576804)
    
    
    
    
    count_SSEWMA = 0
   
    
    # Count number of observations before OOC
    for jj in range(start_OOC,run_length,1):
        if result_SSEWMA[val2]['state'][jj] == "OOC" and count_SSEWMA == 0:
            count_SSEWMA += jj - start_OOC + 1
            break
  
    # print(count_SSEWMA)
    # plt.style.use('science')
    # plt.figure()
    # plt.rc('font', size=14)
    # plt.rc('axes', titlesize=18)
    # plt.rc('axes', labelsize=18)
    # plt.rc('xtick', labelsize=18)
    # plt.rc('ytick', labelsize=18)
    # plt.rc('legend', fontsize=18)
    # plt.rc('figure', titlesize=16)
    # plt.grid()
    # plt.plot(result_SSEWMA[val2]['Norm'], label = '$||M_n||^2$')
    # plt.plot(result_SSEWMA[val2]['Limit'], label = r'$LIM_{n,2}$')
    # plt.xlabel('n [-]')
    # plt.ylabel('$||M_n||^2$ [-]')
    # plt.legend()

    ARL_list_SSEWMA.append(count_SSEWMA)
    # Store csv with run results
    # write_to_csv(result_SSEWMA.T, filepath, filename_SSEWMA)
    
    

    return result_SSEWMA, count_SSEWMA

def get_combined_results(count_combi, full_data, filepath, filename_Combi_Q, ARL_list_Combi, clustered_data_DBSCAN, noise_clusters, filename_Combi_SSEWMA):
    # Initialize variables
    # Subtract 1, because we take the length of the df
    # Shift occurs at Observation 20, but we start counting at 0, thus len give 21 instead of 20
    observations = len(full_data.T) - 1
    print(observations)
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
    if noise_clusters[-1].empty == False:
        result_Q_chart = Q_Chart.Calculate_Q_Chart_UCL_LCL(noise_clusters[-1], 1)
    else:
        result_Q_chart = pd.DataFrame()
    
    
    # Detect special cause variation
    if val2 in result_SSEWMA:
        # Count number of observations before OOC
        if result_SSEWMA[val2]['state'][-1] == "OOC" and count_combi == 0:
            count_combi = observations - start_OOC + 1
        

           
    if res in result_Q_chart:
        # Count number of observations before OOC
        if result_Q_chart[res]['States'][-1] == "OOC" and count_combi == 0:
            count_combi = observations - start_OOC + 1
        
    # if count_combi != 0:
    #     print(count_combi)
    #     plt.figure()
    #     plt.style.use('science')
    #     plt.grid()
    #     plt.plot(result_SSEWMA[val2]['Norm'], label = '$||M||^2$')
    #     plt.plot(result_SSEWMA[val2]['Limit'], label = 'Limit (h)')
    #     plt.xlabel('Observations [-]')
    #     plt.ylabel('$||M||^2$')
    #     plt.legend()
        
        # plt.figure()
        # plt.plot(result_Q_chart[res]['Observation'], label = 'Observation')
        # plt.plot(result_Q_chart[res]['LCL'], c = 'r', label = 'Control limit')
        # plt.plot(result_Q_chart[res]['UCL'], c = 'r')
        # plt.legend()
        # plt.ylabel(f"{res}")


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
       
        
    # print(count_CP)
    # plt.figure()
    # plt.rc('font', size=14)
    # plt.rc('axes', titlesize=18)
    # plt.rc('axes', labelsize=18)
    # plt.rc('xtick', labelsize=18)
    # plt.rc('ytick', labelsize=18)
    # plt.rc('legend', fontsize=18)
    # plt.grid()
    # plt.plot(result_CP['Z'], label = '$Z_{max,n}$')
    # plt.plot(result_CP['Limit'], label = 'Limit (h)', c = 'orange')
    # plt.xlabel('n [-]')
    # plt.ylabel('$Z_{max,n}$')
    # plt.legend()
  
    ARL_list_CP.append(count_CP)
    # Store csv with run results
    # write_to_csv(result_CP.T, filepath, filename_CP)
    
    return result_CP
 
def covariance_to_correlation(cov_matrix):
    """Convert covariance matrix to correlation matrix."""
    # Calculate the standard deviations
    std_dev = np.sqrt(np.diag(cov_matrix))
    
    # Outer product of standard deviations
    outer_std_dev = np.outer(std_dev, std_dev)
    
    # Divide covariance matrix by outer product of standard deviations
    corr_matrix = cov_matrix / outer_std_dev
    
    # Set diagonal to 1
    np.fill_diagonal(corr_matrix, 1)
    
    return corr_matrix   

if __name__ == "__main__":
    
    # Define global parameters
    run_length = 30
    start_OOC = 20
    runs = 200
    cluster_size = 2
    DBSCAN_threshold = 0.8
    
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
    
    cluster_matrix = np.zeros((97,run_length+1))
    
    for shift_size in np.arange(0, 16, 1):
        Shift_Size = shift_size
        
        
        ARL_list_Q = []
        ARL_list_SSEWMA = []
        ARL_list_Combi = []
        ARL_list_CP = []
   
        
        
        
        tot_noise_cluster_number = []
        tot_total_cluster_list = []
        tot_biggest_cluster_list = []
        T_list = []
        data_real, cov = Load_Data(32)
        
        for run in range(runs):
           
            # Define data storage locations
            filepath = r"C:\Users\tbeene\Desktop\Simulation\Full_Simulation\Full_simulation_OOC_Tau_5"
            filename_SSEWMA = f"run_SSMEWMA_{run}_{np.round(shift_size,2)}.csv"
            filename_CP = f"run_MCPD_{run}_{np.round(shift_size,2)}.csv"
            filename_Q = f"run_Q_{run}_{np.round(shift_size,2)}.csv"
            filename_Combi_Q = f"run_Combined_{run}_{np.round(shift_size,2)}_Q.csv"
            filename_Combi_SSEWMA = f"run_Combined_{run}_{np.round(shift_size,2)}_SSEWMA.csv"
            filename_results = f"global_results_{np.round(shift_size,2)}.csv"
            print(f"Simulation percentage complete: ", 100*(run/runs))
            print(f"Shift size: {shift_size}")
            
            
         

            full_data = Generate_SPC_MEZCAL_Data_V1.generate_multivariate_data(data_real, run_length).T

            # Generate a random process reading to be OOC
            val = random.randint(1, len(full_data) - 1)
            res = list(full_data.T.keys())[val]
            
            # Calculate random clusters based on fixed cluster size
            cluster_data_SSEWMA = apply_fixed_clustering(full_data, cluster_size, data_real)
            
            # # Q, SSMEWMA, MCPD results
            results_Q = get_Q_results(full_data.copy(), filepath, filename_Q, ARL_list_Q)
            result_SSEWMA, count_SSEWMA = get_SSEWMA_results(cluster_data_SSEWMA, full_data.copy(), filepath, filename_SSEWMA, ARL_list_SSEWMA)
            results_CP = get_CP_results(full_data.copy(), filepath, filename_CP, ARL_list_CP)
            
           
            # Get combination chart results by reclustering every timestep
            count_combi = 0
            for current_run_length in range(start_OOC,run_length,1):
                print(current_run_length)
                # Speed up simulations by stopping when OOC is detected
                if count_combi != 0:
                    break

                current_data = full_data.copy().iloc[:,:current_run_length+1].T
            
                
                # Recluster with DBSCAN based on current data set
                clustered_data_DBSCAN, cluster_data_noise, cluster_size_counts_dict, T_list = find_groups(current_data, data_real, T_list, cluster_matrix)
                
                
                # Determine the SSEWMA and Q results based on the reclustered data
                results_SSEWMA, result_Q_chart, count_combi = get_combined_results(count_combi, current_data.T, filepath, filename_Combi_Q, ARL_list_Combi, clustered_data_DBSCAN, cluster_data_noise, filename_Combi_SSEWMA)
               
        
                
  
            # ARL_OOC
            ARL_list_Combi.append(count_combi)
            # ARL_list_CP.append(0)
            # Store csv with run results for combined chart
            # write_to_csv(result_SSEWMA.T, filepath, filename_Combi_SSEWMA)
            # write_to_csv(result_Q_chart.T, filepath, filename_Combi_Q)
          
            
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
    
    
        # # CP method
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
    # 95% confidence interval
    
    yerrCombi = 2*1.96*np.sqrt((POD_list_Combi*(1-POD_list_Combi))/len(ARL_list_Combi))
    yerrCombi2 = 2*1.96*np.sqrt((POD2_list_Combi*(1-POD2_list_Combi))/len(ARL_list_Combi))
    yerrCombi5 = 2*1.96*np.sqrt((POD5_list_Combi*(1-POD5_list_Combi))/len(ARL_list_Combi))
       

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
    yerrSSEWMA = 2*1.96*np.sqrt((POD_list_SSEWMA*(1-POD_list_SSEWMA))/len(ARL_list_SSEWMA))
    yerrSSEWMA2 = 2*1.96*np.sqrt((POD2_list_SSEWMA*(1-POD2_list_SSEWMA))/len(ARL_list_SSEWMA))
    yerrSSEWMA5 = 2*1.96*np.sqrt((POD5_list_SSEWMA*(1-POD5_list_SSEWMA))/len(ARL_list_SSEWMA))
       

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
    yerrQ = 2*1.96*np.sqrt((POD_list_Q*(1-POD_list_Q))/len(ARL_list_Q))
    yerrQ2 = 2*1.96*np.sqrt((POD2_list_Q*(1-POD2_list_Q))/len(ARL_list_Q))
    yerrQ5 = 2*1.96*np.sqrt((POD5_list_Q*(1-POD5_list_Q))/len(ARL_list_Q))
       

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
    yerrCP = 2*1.96*np.sqrt((POD_list_CP*(1-POD_list_CP))/len(ARL_list_CP))
    yerrCP2 = 2*1.96*np.sqrt((POD2_list_CP*(1-POD2_list_CP))/len(ARL_list_CP))
    yerrCP5 = 2*1.96*np.sqrt((POD5_list_CP*(1-POD5_list_CP))/len(ARL_list_CP))
       

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




plt.style.use('science')

shift_size = np.arange(1, 6, 1)

fig, axs = plt.subplots(3, 1, figsize=(7, 12), sharex=True)

# --- Colors and markers for consistency ---
methods = [
    ('P-SSMEWMA', 'g--o', 'black'),
    ('R-SSMEWMA', 'b--o', 'black'),
    ('Q', 'r--o', 'black'),
    ('HC', 'y--o', 'black')
    
]

# --- Data for each POD level ---
pod_levels = {
    2: [POD2_list_combi, POD2_list_SSEWMA, POD2_list_Q, POD2_list_CP],
    5: [POD5_list_combi, POD5_list_SSEWMA, POD5_list_Q, POD5_list_CP],
    10: [POD_list_combi, POD_list_SSEWMA, POD_list_Q, POD_list_CP],
}
yerrs = {
    2: [yerrCombi2, yerrSSEWMA2, yerrQ2, yerrCP2],
    5: [yerrCombi5, yerrSSEWMA5, yerrQ5, yerrCP5],
    10: [yerrCombi, yerrSSEWMA, yerrQ, yerrCP],
}

# Set font sizes
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)   # fontsize of the figure title

for idx, pod_level in enumerate([2, 5, 10]):
    ax = axs[idx]
    pod_data = pod_levels[pod_level]
    pod_errs = yerrs[pod_level]

    for method_idx, (label_template, fmt, ecolor) in enumerate(methods):
        ax.errorbar(
            shift_size,
            pod_data[method_idx],
            yerr=pod_errs[method_idx],
            capsize=3,
            fmt=fmt,
            ecolor=ecolor,
            label=label_template
        )

    ax.set_ylabel(f"$POD_{{{pod_level}}}$ [-]")
    ax.grid(True)
    
axs[0].legend(loc='upper left')
axs[0].set_xticks(shift_size)

axs[2].set_xlabel(r'$\delta$ [-]')

plt.tight_layout()
plt.savefig('POD_2_5_10_AllMethods.pdf')
plt.show()


