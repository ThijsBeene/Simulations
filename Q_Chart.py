from scipy.special import stdtrit # Inverse cdf of student t
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_dist_values(N, k):
    """This function calculates the statistics for the Q-chart."""
    df = N - 1
    normal_cdf = stats.norm.cdf(k) # Adjusting to use k instead of fixed 3
    t_cdf = stdtrit(df, normal_cdf)
    value = t_cdf * ((N + 1) / N) ** 0.5
    return value


def Q_Chart(univariate_data, N, k):
    """Computes UCL and LCL for the Q-chart."""
    mean = np.mean(univariate_data)
    std_dev = np.std(univariate_data, ddof=1)
    Q_Calc = calc_dist_values(N, k)
    UCL = mean + Q_Calc * std_dev
    LCL = mean - Q_Calc * std_dev
    return UCL, LCL

def Check_State(current_observation, UCL, LCL):
    """Checks if the current observation is in-control (IC) or out-of-control (OOC)."""
    if (UCL is not None and current_observation > UCL) or (LCL is not None and current_observation < LCL):
        return "OOC"
    return "IC"

def Calculate_Q_Chart_UCL_LCL(full_data, k):
    """Computes the Q-chart UCL, LCL, and states for the data."""
    result_df = pd.DataFrame()

    for key, value in full_data.T.items():
        UCL_list, LCL_list, state_list, include_list = [], [], [], []
       
        for ii in range(len(value)):
            current_observation = value[ii]
            
            n = len(include_list)
            if n >= 6: # Q-chart requires at least 6 points
                UCL, LCL = Q_Chart(include_list, n, k)
            else:
                UCL, LCL = None, None

            state = Check_State(current_observation, UCL, LCL)

            if state == "IC":
                include_list.append(current_observation) # Only add if it is in control
            
            state_list.append(state)
            UCL_list.append(UCL)
            LCL_list.append(LCL)

        result_df[key] = {"Observation": value.tolist(), "UCL": UCL_list, "LCL": LCL_list, "States": state_list}

    return result_df
