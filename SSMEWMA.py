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
            if i >= j and np.linalg.det(X_prev.T @ X_prev) != 0:
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

def Calculate_SSEWMA_Norm_Lim(cluster_data, h_type=0):
    
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
            h = 1*np.array([0, 0, 8.458, 10.574, 12.484, 14.274,
                  22.09700000000012, 17.581, 19.165, 20.732, 
                  22.302, 23.335297671491503, 24.873681246066706, 26.412064820641913, 27.950448395217116, 29.534, 31.027215544367525, 32.56559911894273, 34.103982693517935, 35.64236626809314, 36.422, 38.71913341724355, 40.257516991818754, 41.79590056639396, 43.33428414096916, 44.87266771554437, 46.41105129011957, 47.94943486469478, 49.48781843926998, 51.026202013845186, 52.56458558842039, 54.10296916299559, 55.6413527375708, 57.179736312146005, 58.718119886721205, 60.25650346129641, 61.79488703587162, 63.33327061044682, 64.87165418502202, 66.41003775959723, 67.94842133417244, 69.48680490874763, 71.02518848332284, 72.56357205789804, 74.10195563247325, 75.64033920704846, 77.17872278162366, 78.71710635619887, 80.25548993077406, 81.79387350534927, 83.33225707992447, 84.87064065449968, 86.40902422907489, 87.9474078036501, 89.4857913782253, 91.0241749528005, 92.5625585273757, 94.1009421019509, 95.63932567652611, 97.17770925110132, 98.71609282567653, 100.25447640025172, 101.79285997482692, 103.33124354940213, 104.86962712397734, 106.40801069855254, 107.94639427312775, 109.48477784770296, 111.02316142227815, 112.56154499685336, 114.09992857142856, 115.63831214600377, 117.17669572057898, 118.71507929515418, 120.25346286972938, 121.79184644430458, 123.33023001887979, 124.868613593455, 126.4069971680302, 127.94538074260541, 129.4837643171806, 131.0221478917558, 132.560531466331, 134.0989150409062, 135.6372986154814, 137.17568219005662, 138.7140657646318, 140.25244933920703, 141.79083291378222, 143.32921648835745, 144.86760006293264, 146.40598363750783, 147.94436721208305, 149.48275078665824, 151.02113436123346, 152.55951793580866, 154.09790151038385, 155.63628508495907, 157.17466865953426, 158.71305223410948])[n] # Linear fitted values! Not very accurate!
        
        else:
            h = 1.66*np.array([0, 0, 8.458, 10.574, 12.484, 14.274,
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



