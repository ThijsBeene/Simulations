# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:49:56 2024

@author: tbeene
"""
from scipy.special import stdtrit # Inverse cdf of student t
from scipy import stats
import numpy as np
import pandas as pd

def calc_dist_values(N, k = 3.891):
    """This function calculates the statistics for the Q-chart
    k is set to a default of 2.74. This value of k ensures that the control limits will
    equal the control limits as set by a Shewhart I chart at N = 30
    """
    # Define df
    df = N - 1

    # Calculate the CDF of the standard normal distribution for the given t_cdf value
    normal_cdf = stats.norm.cdf(k)
    # Calculate the inverse cdf of the student t distribution of the normal
    t_cdf = stdtrit(df, normal_cdf)

    value = t_cdf * ((N+1)/N)**0.5

    return value


#Define Q_Chart
def Q_Chart(univariate_data, N, k):
    # N is the number of observations
    mean = np.mean(univariate_data)
    std_dev = np.std(univariate_data) if len(univariate_data) > 1 else 0 # Std dev is 0 for single value
    
    # Initialize LCL and UCL to None
    LCL, UCL = None, None
    
    # Calculate LCL and UCL if N is 6 or greater
    if N >= 6:

        # Standard values for first 30 observations monitored by Q-chart
        Q_Calc = calc_dist_values(N, k)
        
        
        UCL = mean + Q_Calc * std_dev
        LCL = mean - Q_Calc * std_dev


    return UCL, LCL


def Check_State(current_observation, UCL, LCL, index):
    # Check if observation is IC or OOC
    if (UCL is not None and current_observation > UCL) or (LCL is not None and current_observation < LCL):
        state_new = "OOC"
    else:
        state_new = "IC"

    return state_new

def Calculate_Q_Chart_UCL_LCL(full_data, k):
    result_df = pd.DataFrame()
    for key, value in full_data.T.items():
        UCL_list = []
        LCL_list = []
        state_list = []
        include_list = []
        for ii in range(0, len(value), 1):
            
            

            running_list = list(value)[:ii]
            current_observation = list(value)[ii]
            
            if ii >= 5:
                UCL, LCL = Q_Chart(include_list, ii, k)
            else:
                UCL, LCL = None, None
            
            state = Check_State(current_observation, UCL, LCL, ii)

            if state == "IC":
                include_list.append(current_observation)
                
            state_list.append(state)
            UCL_list.append(UCL)
            LCL_list.append(LCL)
            
        result_df[key] = {"Observation": running_list, "UCL": UCL_list, "LCL": LCL_list, "States": state_list}
        
        
    return result_df
        
    