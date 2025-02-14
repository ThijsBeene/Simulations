# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:19:22 2024

@author: tbeene
"""

from scipy.linalg import qr, solve_triangular
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
    Compute the recursive residuals matrix R using QR factorization, following the approach in Hawkins (2007). 
    """ 
    X = np.array(X, dtype=np.float64)
    n, p = X.shape 
    R = np.full((n, p), np.nan) # Initialize R with NaNs 
    X = np.hstack((np.ones((n, 1)), X)) # Add intercept column 
    for j in range(1, p+1): # Loop over each variable 
        for i in range(j, n): # Start from the second available observation 
            # Define the regression dataset 
            y = X[:i, j] # Current target variable (column j) 
            y_act = X[i, j] # Actual value for current observation 
            # QR factorization instead of Cholesky 
            Q, R_qr = qr(X[:i, :j], mode='economic') # Compute QR decomposition 
            beta = solve_triangular(R_qr, Q.T @ y, lower = False) # Solve for regression coefficients 
         
            y_pred = X[i, :j] @ beta # Compute predicted value 
            residual = y_act - y_pred # Compute residual 
           
            # Compute leverage using QR 
            h_ij = X[i, :j] @ np.linalg.inv(X[:i, :j].T @ X[:i, :j]) @ X[i, :j]
            # Normalize recursive residual 
            R[i, j-1] = residual / np.sqrt(1 + h_ij) 
            
    # R[5][4] = -0.002
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
    for j in range(p):
        for i in range(j+1,n):
            # Only compute for defined residuals, i.e. i > j
            df = i - j - 1 # degrees of freedom according to the paper
            # Only transform if df > 0 (i.e. at least 3 observations in the regression)
            if df > 0:
                t_n_i = R[i, j]/np.sqrt(np.sum((R[j+1:i, j]**2)/(df)))
                # Compute the cumulative probability of the t-statistic
                p_val = t.cdf(t_n_i, df)
                # Transform to the corresponding quantile of the standard normal
                U[i, j] = norm.ppf(p_val)
               
    return U

def calc_M(U, Lambda):
    """ Compute EWMA statistic. """
    U[np.isnan(U)] = 0
    o, p = U.shape
    M_list = np.zeros_like(U)
    for n in range(0, len(U)):
        if n >= p+1:
            M_list[n] = Lambda * U[n] + (1 - Lambda) * M_list[n-1]
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
        
            

            lim_list.append(0)


    
    return T2_list, lim_list

def Calculate_SSEWMA_Norm_Lim(cluster_data, h_type=0):
    
    index = 0
    result_df = pd.DataFrame()
    
    k = len(cluster_data)

 
    
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


        # Only the first 10, 15 and 20 values are correct and from the paper, the rest was linearly extrapolated
        if h_type == 0: 
            # n =2 should take the first index!
            # We only need to calculate values for the correct cluster siozes: [2,3,4,6,8,12,16,24]
            h = [7.25, 16.29999999999997, 17.99999999999996, 0, 20.75000000000004, 0, 23.300000000000047, 0, 0, 0, 27.750000000000068, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 100][n-2]
            print(h)
        else:
            if n - 2 < 14 and k - 1 < 19:
                # Add rows to the h matrix to increase accuracy!
                h = [[8.79999999999999, 10.099999999999985, 10.899999999999983, 11.49999999999998, 11.899999999999979, 12.299999999999978, 12.599999999999977, 12.799999999999976, 12.999999999999975, 13.299999999999974, 13.399999999999974, 13.599999999999973, 13.699999999999973, 13.899999999999972, 14.099999999999971, 14.19999999999997, 14.29999999999997, 14.39999999999997, 14.49999999999997], 
                 [10.699999999999983, 12.199999999999978, 12.999999999999975, 13.599999999999973, 14.099999999999971, 14.49999999999997, 14.899999999999968, 15.199999999999967, 15.499999999999966, 15.699999999999966, 15.899999999999965, 16.099999999999966, 16.29999999999996, 16.499999999999964, 16.69999999999996, 16.899999999999963, 16.99999999999996, 17.19999999999996, 17.39999999999996],
                 [12.499999999999977, 13.999999999999972, 14.999999999999968, 15.599999999999966, 16.099999999999966, 16.599999999999962, 16.899999999999963, 17.19999999999996, 17.499999999999957, 17.69999999999996, 17.899999999999956, 18.09999999999996, 18.199999999999957, 18.399999999999956, 18.499999999999957, 18.599999999999955, 18.799999999999955, 18.899999999999956, 18.999999999999954],
                 [14.099999999999971, 15.699999999999966, 16.599999999999962, 17.19999999999996, 17.69999999999996, 18.09999999999996, 18.399999999999956, 18.699999999999953, 18.999999999999954, 19.199999999999953, 19.399999999999952, 19.699999999999953, 19.89999999999995, 20.09999999999995, 20.19999999999995, 20.39999999999995, 20.599999999999948, 20.699999999999946, 20.799999999999947], 
                 [15.69999999999999, 17.399999999999984, 18.49999999999998, 19.199999999999978, 19.699999999999974, 20.099999999999973, 20.49999999999997, 20.799999999999972, 21.099999999999973, 21.29999999999997, 21.49999999999997, 21.79999999999997, 21.89999999999997, 22.099999999999966, 22.29999999999997, 22.499999999999964, 22.599999999999966, 22.799999999999965, 22.899999999999963], 
                 [17.099999999999987, 18.899999999999977, 19.899999999999977, 20.599999999999973, 21.099999999999973, 21.59999999999997, 21.89999999999997, 22.199999999999967, 22.499999999999964, 22.799999999999965, 22.999999999999964, 23.199999999999964, 23.399999999999963, 23.599999999999962, 23.79999999999996, 23.99999999999996, 24.09999999999996, 24.19999999999996, 24.29999999999996],
                 [18.49999999999998, 20.399999999999974, 21.49999999999997, 22.29999999999997, 22.799999999999965, 23.29999999999996, 23.69999999999996, 23.99999999999996, 24.29999999999996, 24.59999999999996, 24.799999999999958, 25.09999999999996, 25.299999999999955, 25.499999999999957, 25.699999999999953, 25.799999999999955, 25.999999999999954, 26.09999999999995, 26.199999999999953], 
                 [19.799999999999976, 21.699999999999967, 22.699999999999967, 23.499999999999964, 24.09999999999996, 24.59999999999996, 25.09999999999996, 25.399999999999956, 25.699999999999953, 25.899999999999956, 26.09999999999995, 26.299999999999955, 26.49999999999995, 26.699999999999953, 26.89999999999995, 27.09999999999995, 27.299999999999947, 27.49999999999995, 27.599999999999948], 
                 [21.100000000000044, 23.100000000000072, 24.30000000000009, 25.1000000000001, 25.70000000000011, 26.200000000000117, 26.700000000000124, 27.000000000000128, 27.400000000000134, 27.600000000000136, 27.90000000000014, 28.100000000000144, 28.300000000000146, 28.50000000000015, 28.60000000000015, 28.800000000000153, 28.900000000000155, 29.100000000000158, 29.20000000000016],  
                 [22.400000000000063, 24.500000000000092, 25.70000000000011, 26.50000000000012, 27.10000000000013, 27.600000000000136, 28.000000000000142, 28.300000000000146, 28.60000000000015, 28.900000000000155, 29.20000000000016, 29.400000000000162, 29.600000000000165, 29.800000000000168, 30.00000000000017, 30.200000000000173, 30.300000000000175, 30.500000000000178, 30.60000000000018],
                 [23.500000000000078, 25.600000000000108, 26.800000000000125, 27.600000000000136, 28.200000000000145, 28.800000000000153, 29.30000000000016, 29.600000000000165, 30.00000000000017, 30.300000000000175, 30.60000000000018, 30.70000000000018, 30.900000000000183, 31.100000000000186, 31.30000000000019, 31.500000000000192, 31.700000000000195, 31.900000000000198, 32.1000000000002],
                 [24.70000000000004, 26.90000000000007, 28.100000000000087, 28.900000000000098, 29.70000000000011, 30.200000000000117, 30.600000000000122, 31.000000000000128, 31.300000000000132, 31.700000000000138, 31.90000000000014, 32.200000000000145, 32.40000000000015, 32.70000000000015, 32.900000000000155, 33.000000000000156, 33.20000000000016, 33.30000000000016, 33.50000000000016],
                 [25.700000000000053, 28.100000000000087, 29.300000000000104, 30.200000000000117, 30.900000000000126, 31.400000000000134, 31.90000000000014, 32.300000000000146, 32.60000000000015, 32.900000000000155, 33.20000000000016, 33.40000000000016, 33.600000000000165, 33.90000000000017, 34.10000000000017, 34.20000000000017, 34.400000000000176, 34.60000000000018, 34.80000000000018],
                 [26.800000000000068, 29.1000000000001, 30.40000000000012, 31.300000000000132, 32.10000000000014, 32.60000000000015, 33.10000000000016, 33.50000000000016, 33.80000000000017, 34.10000000000017, 34.400000000000176, 34.60000000000018, 34.80000000000018, 35.000000000000185, 35.20000000000019, 35.40000000000019, 35.50000000000019, 35.60000000000019, 35.800000000000196]][n-2][k-1]
            
            else:
                h = 100
 
    
    
 
    
        
        
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



