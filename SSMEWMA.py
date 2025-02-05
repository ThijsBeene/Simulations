# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:19:22 2024

@author: tbeene
"""


import numpy as np 
import pandas as pd 
from scipy.special import stdtr # Inverse CDF of Student's t-distribution 
from scipy.stats import norm 
from numpy import linalg as LA 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from numpy.linalg import cholesky, inv



# ---- STEP 1: LOAD DATA ----
def generate_multivariate_data(cov, data, run_length):
    """ Generate synthetic multivariate normal data. """
    means = np.mean(data, axis=0)
    samples = np.random.multivariate_normal(mean=means, cov=cov, size=run_length)
    return pd.DataFrame(samples, columns=data.columns)

# ---- STEP 1: LOAD DATA ----
def generate_multivariate_data(cov, data, run_length):
    """ Generate synthetic multivariate normal data. """
    means = np.mean(data, axis=0)
    samples = np.random.multivariate_normal(mean=means, cov=cov, size=run_length)
    return pd.DataFrame(samples, columns=data.columns)

def compute_recursive_residuals(X):
    """
    Compute the recursive residuals matrix R using the approach from the Hawkins (2007) paper.

    Parameters:
        X (numpy.ndarray): n x p matrix of process readings.

    Returns:
        R (numpy.ndarray): (n-p) x p matrix of recursive residuals.
    """
    
    n, p = X.shape
    R = np.full((n, p), np.nan) # Initialize R with NaNs for clarity
    X = X.values
    X = np.hstack((np.ones((n, 1)), X[:, :]))


    for j in range(1,p+1): # Loop over each variable
        for i in range(j, n): # Start from the second available observation
            
            # Define the regression dataset
            X_prev = X[:i, :j] # Past observations for predictors
       
            y = X[:i, j] # Current target variable (column j)

            y_act = X[i, j] # Actual value for current observation
            
            if i >= j:
                # R_CHOL = cholesky(X_prev.T @ X_prev) # Cholesky decomposition
                # beta = inv(R_CHOL.T) @ inv(R_CHOL) @ (X_prev.T @ y)
                # y_pred = X[i, :j] @ beta

                # Solve least squares to get regression coefficients
                beta = np.linalg.lstsq(X_prev, y, rcond=None)[0]
                y_pred = X[i, :j] @ beta # One-step-ahead prediction
                residual = y_act - y_pred # Compute residual
            else:
                residual = y_act - np.mean(y) # First variable case (mean-based adjustment)

            # Compute leverage only if regression is defined
            if i >= j:
                leverage = X[i, :j] @ inv(X_prev.T @ X_prev) @ X[i, :j].T
            else:
                leverage = 0 # No leverage adjustment in the first variable case

            # Normalize recursive residual
            R[i, j-1] = residual / np.sqrt(1 + leverage)

    return R


def transform_to_U(R):
    """
    Transform a matrix of recursive residual t-statistics into independent standard normal variates.
    
    For each valid element R[i, j] (with i > j), we assume the degrees of freedom is:
         df = i - j - 1
    and define the corresponding standard normal variate as:
         u = norm.ppf( t.cdf(R[i,j], df) ).
    
    Parameters:
        R (numpy.ndarray): 2D array of recursive residual t-statistics.
                           (Entries with i <= j are not defined.)
    
    Returns:
        U (numpy.ndarray): 2D array of the same shape as R containing the transformed
                           standard normal variates.
    """
    U = np.full(R.shape, np.nan)
    n, p = R.shape
    for i in range(0,n):
        for j in range(p):
            # Only compute for defined residuals, i.e. i > j
            if i > j+1:
                df = i - j - 1 # degrees of freedom according to the paper
                # Only transform if df > 0 (i.e. at least 3 observations in the regression)
                if df > 0:
                    t_n_i = R[i, j]/np.sqrt(np.sum((R[j+1:i, j]**2)/(df)))
                    # Compute the cumulative probability of the t-statistic
                    p_val = t.cdf(t_n_i, df)
                    # Transform to the corresponding quantile of the standard normal
                    U[i, j] = norm.ppf(p_val)
                else:
                    U[i, j] = R[i, j] # For df==0, simply pass through (should not happen)
    return U

def calc_M(U, Lambda):
    """ Compute EWMA statistic. """
    U[np.isnan(U)] = 0
    o, p = U.shape
    M_list = np.zeros_like(U)
    for i in range(p+1, len(U)):

        M_list[i] = Lambda * U[i] + (1 - Lambda) * M_list[i-1]
    return M_list

def calc_T2_and_limits(M, Lambda, h):
    """ Compute T² statistics and corresponding control limits. """
    T2_list = []
    lim_list = []
    o, p = M.shape


    for n in range(0, len(M)):
        if n >= p+1:
            T2 = np.linalg.norm(M[n])**2
            T2_list.append(T2)

           
            limit = (Lambda * (1 - (1 - Lambda)**(2 * (n - p + 1))) / (2 - Lambda)) * h
            lim_list.append(limit)
        else:
            T2_list.append(0)
        
            
            if n != p:
                lim_list.append(0)


    
    return T2_list, lim_list

def Calculate_SSEWMA_Norm_Lim(cluster_data, sim, h_type=0):
    
    index = 0
    result_df = pd.DataFrame()
    
    for key, cluster in cluster_data.items():
        
        cluster_trans = cluster.copy().T
        columns = cluster.columns
       
        
        T2_list_clusters = []
        lim_list_clusters = []


        n = len(cluster)
        i = len(cluster.columns)
       
       
        
        # Dimension = 7, Lambda = 0.25 and h = 19.534 should result in an ARL_IC of 200
        # A in control process reading with a length of 500 observations would get the ARL_IC within 10% of its asymptotic value
        Lambda = 0.25
        
        # Autmoatically select the correct h value based on cluster size

     
     
        # Only the first 10, 15 and 20 values are correct and from the paper, the rest was linealry extrapolated
        if h_type == 0: 
            h = 1.4*np.array([0, 0, 8.458, 10.574, 12.484, 14.274,
                  15.934, 17.581, 19.165, 20.732, 
                  22.302, 23.335297671491503, 24.873681246066706, 26.412064820641913, 27.950448395217116, 29.534, 31.027215544367525, 32.56559911894273, 34.103982693517935, 35.64236626809314, 36.422, 38.71913341724355, 40.257516991818754, 41.79590056639396, 43.33428414096916, 44.87266771554437, 46.41105129011957, 47.94943486469478, 49.48781843926998, 51.026202013845186, 52.56458558842039, 54.10296916299559, 55.6413527375708, 57.179736312146005, 58.718119886721205, 60.25650346129641, 61.79488703587162, 63.33327061044682, 64.87165418502202, 66.41003775959723, 67.94842133417244, 69.48680490874763, 71.02518848332284, 72.56357205789804, 74.10195563247325, 75.64033920704846, 77.17872278162366, 78.71710635619887, 80.25548993077406, 81.79387350534927, 83.33225707992447, 84.87064065449968, 86.40902422907489, 87.9474078036501, 89.4857913782253, 91.0241749528005, 92.5625585273757, 94.1009421019509, 95.63932567652611, 97.17770925110132, 98.71609282567653, 100.25447640025172, 101.79285997482692, 103.33124354940213, 104.86962712397734, 106.40801069855254, 107.94639427312775, 109.48477784770296, 111.02316142227815, 112.56154499685336, 114.09992857142856, 115.63831214600377, 117.17669572057898, 118.71507929515418, 120.25346286972938, 121.79184644430458, 123.33023001887979, 124.868613593455, 126.4069971680302, 127.94538074260541, 129.4837643171806, 131.0221478917558, 132.560531466331, 134.0989150409062, 135.6372986154814, 137.17568219005662, 138.7140657646318, 140.25244933920703, 141.79083291378222, 143.32921648835745, 144.86760006293264, 146.40598363750783, 147.94436721208305, 149.48275078665824, 151.02113436123346, 152.55951793580866, 154.09790151038385, 155.63628508495907, 157.17466865953426, 158.71305223410948])[n] # Linear fitted values! Not very accurate!
        
        else:
            h = 1.68*np.array([0, 0, 8.458, 10.574, 12.484, 14.274,
                  15.934, 17.581, 19.165, 20.732, 
                  22.302, 23.335297671491503, 24.873681246066706, 26.412064820641913, 27.950448395217116, 29.534, 31.027215544367525, 32.56559911894273, 34.103982693517935, 35.64236626809314, 36.422, 38.71913341724355, 40.257516991818754, 41.79590056639396, 43.33428414096916, 44.87266771554437, 46.41105129011957, 47.94943486469478, 49.48781843926998, 51.026202013845186, 52.56458558842039, 54.10296916299559, 55.6413527375708, 57.179736312146005, 58.718119886721205, 60.25650346129641, 61.79488703587162, 63.33327061044682, 64.87165418502202, 66.41003775959723, 67.94842133417244, 69.48680490874763, 71.02518848332284, 72.56357205789804, 74.10195563247325, 75.64033920704846, 77.17872278162366, 78.71710635619887, 80.25548993077406, 81.79387350534927, 83.33225707992447, 84.87064065449968, 86.40902422907489, 87.9474078036501, 89.4857913782253, 91.0241749528005, 92.5625585273757, 94.1009421019509, 95.63932567652611, 97.17770925110132, 98.71609282567653, 100.25447640025172, 101.79285997482692, 103.33124354940213, 104.86962712397734, 106.40801069855254, 107.94639427312775, 109.48477784770296, 111.02316142227815, 112.56154499685336, 114.09992857142856, 115.63831214600377, 117.17669572057898, 118.71507929515418, 120.25346286972938, 121.79184644430458, 123.33023001887979, 124.868613593455, 126.4069971680302, 127.94538074260541, 129.4837643171806, 131.0221478917558, 132.560531466331, 134.0989150409062, 135.6372986154814, 137.17568219005662, 138.7140657646318, 140.25244933920703, 141.79083291378222, 143.32921648835745, 144.86760006293264, 146.40598363750783, 147.94436721208305, 149.48275078665824, 151.02113436123346, 152.55951793580866, 154.09790151038385, 155.63628508495907, 157.17466865953426, 158.71305223410948])[n] # Linear fitted values! Not very accurate!
        
        
        
        R = compute_recursive_residuals(cluster_trans)
        U = transform_to_U(R)
        M = calc_M(U, Lambda)
        T2_list, lim_list = calc_T2_and_limits(M, Lambda, h)
        
 

        # Check if UCL has been exceeded
        count = 0
        state = []
        for x,y in zip(T2_list, lim_list):
            
            if x > y:
                state.append("OOC")
                count += 1
            else:
                state.append("IC")
       

        result_df[index] = {"Norm" : T2_list, "Limit" : lim_list, "Num_OOC": count, "state": state, "h": h}
            
        index += 1
           
            
            
    return result_df


# Old code!
# -*- coding: utf-8 -*-
# """
# Created on Mon Dec 16 09:19:22 2024

# @author: tbeene
# """

# import numpy as np
# import pandas as pd
# from scipy.special import stdtr # Inverse cdf of student t
# from scipy.stats import norm
# from numpy import linalg as LA



# def find_residuals(df, i):
    
#     # The recursive residual r_t_i can be found by regression fitting all past observations
#     # And calculating the difference between the linear regression prediction and the actual value
#     df_old = np.array(df.copy().iloc[:,:-1])
    
#     final_x = np.array(df.iloc[:,-1]).reshape(i)
#     x_predict = np.array([])
#     # Loop over the process reading vectors
#     for row in df_old:

#         x = np.arange(1, len(row) + 1, 1)
#         x_next_index = len(row) + 1
 
#         m,b = np.polyfit(x, np.array(row), 1)
        
#         x_next_predict = m*x_next_index + b #np.mean(row)#m*x_next_index + b
#         x_predict = np.append(x_predict, x_next_predict)

#     # Return residual vector r_t_i
#     return final_x - x_predict


# def calc_old_res(df_curr, n, index):
#     normalized = 0

#     for ii in range(index+1, n-1, 1):
        
#         val = find_residuals(df_curr.iloc[:,:ii + 1], index)
      
#         normalized += (val**2)/(n - index - 1 - 1) # Subtract the extra 1 that was added to compensate for python counting?
        

        
#     return normalized

# def calc_U(df_random, Lambda, n, i):
#     U = []
#     for t in range(1, n + 1, 1):
#         if t > i + 2:
#             df_curr = df_random.copy().iloc[:,:t + 1]
            
            
#             normalized = calc_old_res(df_curr, t, i)

   
            
#             res_cur = find_residuals(df_curr, i)
        
      
#             t_res = [res_cur[l]/(normalized[l]**0.5) for l in range(len(res_cur))] 
            
#             # Calculate the probability integral transform
#             # Define df
#             df = t - i - 1 - 1 # Subtract the extra 1 that was added to compensate for python counting?
#             t_cdf = stdtr(df, t_res)
            
#             # Calculate the CDF of the standard normal distribution for the given t_cdf value
#             normal_cdf_inv = norm.ppf(t_cdf)
            
#             U.append(normal_cdf_inv)
#         else:
#             U.append([0]*i)
            
#     return U
    



# def calc_M(U, Lambda, n, i):
#     M_new_list = []
#     # Initialize list with 0: M_0 = 0
#     M_old = [0]
    
#     for p in range(len(U)): 
#         M_new = Lambda*U[p] + (1 - Lambda) * M_old[p]
        
#         M_old.append(M_new)
        
#         M_new_list.append(M_new)
        
#     return M_new_list




# def calc_t2_lim(M_new_list, Lambda, n, i, h):
#     T2_list = [0]
#     lim_list = [0]


#     #for M in M_new_list:
#     for n in range(0, len(M_new_list), 1):
#         if n > i + 1:   
#             T2 = ((2-Lambda)/(Lambda*(1-(1-Lambda)**(2*(n - i - 1)))))*(LA.norm(M_new_list[n]))**2
#             lim = (Lambda*(1 - (1 - Lambda)**(2*(n - i - 1)))/(2 - Lambda)) * h

            
#             lim_list.append(lim)
#             T2_list.append(T2)
            
#         else:
#             lim_list.append(0)
#             T2_list.append(0)
      
            
#     return T2_list, lim_list


# def Calculate_SSEWMA_Norm_Lim(cluster_data, sim, h_type=0):
    
#     index = 0
#     result_df = pd.DataFrame()
    
#     for cluster in cluster_data.values():
#         print(cluster)
        
#         columns = cluster.T.columns
       
        
#         T2_list_clusters = []
#         lim_list_clusters = []


#         n = len(cluster.columns)
#         i = len(cluster)
#         print(i)
        
#         # Dimension = 7, Lambda = 0.25 and h = 19.534 should result in an ARL_IC of 200
#         # A in control process reading with a length of 500 observations would get the ARL_IC within 10% of its asymptotic value
#         Lambda = 0.25
        
#         # Autmoatically select the correct h value based on cluster size

     
     
#         # Only the first 10, 15 and 20 values are correct and from the paper, the rest was linealry extrapolated
#         # if h_type == 1: 
#         #     h = 4.3*np.array([0, 0, 8.458, 10.574, 12.484, 14.274,
#         #           15.934, 17.581, 19.165, 20.732, 
#         #           22.302, 23.335297671491503, 24.873681246066706, 26.412064820641913, 27.950448395217116, 29.534, 31.027215544367525, 32.56559911894273, 34.103982693517935, 35.64236626809314, 36.422, 38.71913341724355, 40.257516991818754, 41.79590056639396, 43.33428414096916, 44.87266771554437, 46.41105129011957, 47.94943486469478, 49.48781843926998, 51.026202013845186, 52.56458558842039, 54.10296916299559, 55.6413527375708, 57.179736312146005, 58.718119886721205, 60.25650346129641, 61.79488703587162, 63.33327061044682, 64.87165418502202, 66.41003775959723, 67.94842133417244, 69.48680490874763, 71.02518848332284, 72.56357205789804, 74.10195563247325, 75.64033920704846, 77.17872278162366, 78.71710635619887, 80.25548993077406, 81.79387350534927, 83.33225707992447, 84.87064065449968, 86.40902422907489, 87.9474078036501, 89.4857913782253, 91.0241749528005, 92.5625585273757, 94.1009421019509, 95.63932567652611, 97.17770925110132, 98.71609282567653, 100.25447640025172, 101.79285997482692, 103.33124354940213, 104.86962712397734, 106.40801069855254, 107.94639427312775, 109.48477784770296, 111.02316142227815, 112.56154499685336, 114.09992857142856, 115.63831214600377, 117.17669572057898, 118.71507929515418, 120.25346286972938, 121.79184644430458, 123.33023001887979, 124.868613593455, 126.4069971680302, 127.94538074260541, 129.4837643171806, 131.0221478917558, 132.560531466331, 134.0989150409062, 135.6372986154814, 137.17568219005662, 138.7140657646318, 140.25244933920703, 141.79083291378222, 143.32921648835745, 144.86760006293264, 146.40598363750783, 147.94436721208305, 149.48275078665824, 151.02113436123346, 152.55951793580866, 154.09790151038385, 155.63628508495907, 157.17466865953426, 158.71305223410948])[i] # Linear fitted values! Not very accurate!
        
#         # else:
#         h = 1.26*np.array([0,0,13.370, 15.832, 18.036, 20.085, 21.967, 23.828, 25.625, 27.372, 29.140, 30.20469225928256, 31.92730522341094, 33.64991818753932, 35.3725311516677, 37.143, 38.81775707992447, 40.54037004405285, 42.26298300818123, 43.98559597230962, 44.730, 47.43082190056638, 49.153434864694766, 50.87604782882315, 52.59866079295153, 54.32127375707991, 56.04388672120829, 57.766499685336676, 59.489112649465056, 61.21172561359344, 62.934338577721824, 64.6569515418502, 66.37956450597858, 68.10217747010697, 69.82479043423534, 71.54740339836373, 73.27001636249211, 74.99262932662049, 76.71524229074888, 78.43785525487725, 80.16046821900564, 81.88308118313402, 83.6056941472624, 85.32830711139079, 87.05092007551917, 88.77353303964755, 90.49614600377593, 92.21875896790431, 93.9413719320327, 95.66398489616108, 97.38659786028946, 99.10921082441784, 100.83182378854622, 102.5544367526746, 104.27704971680299, 105.99966268093137, 107.72227564505975, 109.44488860918814, 111.16750157331651, 112.8901145374449, 114.61272750157329, 116.33534046570166, 118.05795342983005, 119.78056639395842, 121.50317935808681, 123.2257923222152, 124.94840528634357, 126.67101825047196, 128.39363121460033, 130.11624417872872, 131.8388571428571, 133.5614701069855, 135.28408307111388, 137.00669603524227, 138.72930899937063, 140.45192196349902, 142.1745349276274, 143.8971478917558, 145.61976085588415, 147.34237382001254, 149.06498678414093, 150.78759974826932, 152.5102127123977, 154.23282567652606, 155.95543864065445, 157.67805160478284, 159.40066456891122, 161.1232775330396, 162.845890497168, 164.56850346129636, 166.29111642542475, 168.01372938955313, 169.73634235368152, 171.4589553178099, 173.18156828193827, 174.90418124606666, 176.62679421019504, 178.34940717432343, 180.07202013845182, 181.79463310258018])[i]
#             #h = [0, 0, 8.458, 10.574, 12.484, 14.274, 15.934, 17.581, 19.165, 20.732, 22.302][i]
  
        
        
        
        
#         U = calc_U(cluster, Lambda, n, i)
#         U = np.array(U)
        
#         M_new_list = calc_M(U, Lambda, n, i)
        
  
#         T2_list, lim_list = calc_t2_lim(M_new_list, Lambda, n, i, h)
       
#         # Check if UCL has been exceeded
#         count = 0
#         state = []
#         for x,y in zip(T2_list, lim_list):
            
#             if x > y:
#                 state.append("OOC")
#                 count += 1
#             else:
#                 state.append("IC")
       

#         result_df[index] = {"Norm" : T2_list, "Limit" : lim_list, "Num_OOC": count, "state": state, "h": h}
            
#         index += 1
           
            
            
#     return result_df
