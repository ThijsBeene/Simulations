import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from numpy.linalg import inv
from numpy.random import multivariate_normal
from scipy.linalg import qr, solve_triangular
from scipy.stats import t, norm
import numpy as np
from scipy.stats import multivariate_t
import numpy as np
from scipy.special import multigammaln
from scipy.stats import invwishart
from numpy.linalg import det






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
        if n >= p+2:
            T2 = np.linalg.norm(M[n])**2
            T2_list.append(T2)

           
            limit = ((Lambda * (1 - (1 - Lambda)**(2 * (n - p - 1))) / (2 - Lambda))) * h
            lim_list.append(limit)
        else:
            T2_list.append(0)
            lim_list.append(0)


    
    return T2_list, lim_list




# First h values are from Hawkins paper, higher then index 11 is not
h = [8.458, 10.574, 12.484, 14.274, 15.934, 17.581, 19.165, 20.732, 22.302, 24, 26, 28, 30, 31,32,33,34,35,36,37,38,39,40]


import numpy as np
from numpy.linalg import slogdet

def calculate_evidence(X, N0=1):
    """
    Approximate the log marginal likelihood (model evidence) for a multivariate Gaussian
    using a MAP covariance estimator with shrinkage toward the identity matrix.

    Parameters:
    - X: (n, d) data matrix
    - N0: prior strength (controls shrinkage intensity)

    Returns:
    - Approximate log marginal likelihood (per dimension)
    """
    n, d = X.shape

    # Sample covariance matrix
    S = np.cov(X[:-1, :].T)


    # Use slogdet for numerical stability
    sign, logdet = slogdet(S)
    
    # Check for non positive definite matrix
    if sign <= 0:
        logdet = 0

    # Compute log entropy of the Gaussian
    log_entropy = -0.5 * (d * np.log(2 * np.pi) + logdet)

    # Return average log marginal likelihood per dimension
    return -log_entropy 



def generate_multivariate_gaussian(n):
    # Generate random means for n time series
    means = np.array([np.random.uniform(0.0, 1) for _ in range(n)])

    
    # Generate a random covariance matrix for n time series
    A = np.random.rand(n, n)
    covariance_matrix = np.dot(A, A.transpose())
    
    
    # Generate 100 observations for n time series
    observations = np.random.multivariate_normal(means, covariance_matrix, 21)
    
    return means, covariance_matrix, observations




num_samples = 1000
sensor_range = range(2, 11, 1)
Lambda = 0.25
color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'magenta', 'orange'] # 7 colors for n=2 to n=8

sum_corrs_by_n = {n: [] for n in sensor_range}
sum_corrs_by_n_ent = {n: [] for n in sensor_range}
T2_ratios_by_n = {n: [] for n in sensor_range}



shift_dict_y = {i: [] for i in range(0, 18, 2)}
shift_dict_x = {i: [] for i in range(0, 18, 2)}

shift_dict_y = {i: [] for i in range(10, 50, 5)}
shift_dict_x = {i: [] for i in range(10, 50, 5)}

# for shift in np.arange(10, 50, 5):
for _ in range(num_samples):
    n = np.random.choice(sensor_range)
    try:
        # means, corr, X = generate_multivariate_t(n, 21, 4)
        means, corr, X = generate_multivariate_gaussian(n)
        
        
      
        entropy=calculate_evidence(X)


        
        #X = multivariate_normal(mean=np.ones(n), cov=corr, size=21)
        X[20][0] += 10*np.std(X.T[0])
        R = compute_recursive_residuals(X)
        U = transform_to_U(R)
        M = calc_M(U, Lambda)
        T2, LIM = calc_T2_and_limits(M, Lambda, h=h[n-2])

        ratio = T2[20] / LIM[20]

        R[np.isnan(R)] = 0
        
       
    
        limit = Lambda*((1-(1-Lambda)**(2*(20 - n - 1)))*h[n-2])/(2-Lambda)

        sum_corrs_by_n[n].append(entropy)
        
        
        T2_ratios_by_n[n].append(ratio)
    except np.linalg.LinAlgError:
        continue

#     for i, n in enumerate(sensor_range):
#         x = np.array(sum_corrs_by_n[n])
#         y = np.array(T2_ratios_by_n[n])
    
#         # Filter out NaN and Inf values
#         mask = np.isfinite(x) & np.isfinite(y)
#         x_clean = x[mask]
#         y_clean = y[mask]
    
#         shift_dict_y[shift].append(np.mean(y_clean))
#         shift_dict_x[shift].append(np.mean(x_clean))
        
# import scienceplots

# plt.style.use('science')

# color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
# marker_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+']

# plt.figure()

# # Set font sizes
# plt.rc('font', size=12)
# plt.rc('axes', titlesize=14)
# plt.rc('axes', labelsize=16)
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)
# plt.rc('legend', fontsize=12)
# plt.rc('figure', titlesize=16)

# for shift in shift_dict_x:
#     x_vals = shift_dict_x[shift]
#     y_vals = shift_dict_y[shift]

#     if len(x_vals) == len(y_vals): # Sanity check
#         color = color_list[shift % len(color_list)]

#         for i, n in enumerate(sensor_range):
#             marker = marker_list[i % len(marker_list)]

#             # Plot with color and marker
#             # Add label only once per shift to the legend
#             label = r'$\tau = $' + f'{shift}' if i == 0 else None

#             plt.scatter(x_vals[i], y_vals[i], alpha=0.5, edgecolors='k',
#                         color=color, marker= marker, s = 100, label=label)

# plt.legend()
# plt.grid()
# plt.xlabel(r'$E\left(2H - k \right)$')
# plt.ylabel(r'$E\left(\frac{||M_{20}(\delta)||^2}{LIM_{20}}\right)$')
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns




import scienceplots

plt.style.use('science')

sensor_range = range(2,11, 1)
color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
marker_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+']

# Set font sizes
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=16)





# Plotting
for i, n in enumerate(sensor_range):
    x = np.array(sum_corrs_by_n[n])
    y = np.array(T2_ratios_by_n[n])
    x2 = np.array(sum_corrs_by_n_ent[n])
    
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    plt.errorbar(
        np.mean(x_clean), np.mean(y_clean),
        xerr=2*1.96 * np.std(x_clean) / np.sqrt(len(x_clean)),
        yerr=2*1.96 * np.std(y_clean) / np.sqrt(len(y_clean)),
        alpha=0.5,
        color=color_list[i % len(color_list)],
        fmt=marker_list[i % len(marker_list)],
        markersize=12,
        ecolor='black',
        capsize=5,
        label=f'p = {n}'
    )
    
    
    
   

plt.legend()
plt.grid()
plt.xlabel(r'$\bar{\mathcal{L}}$ [-]')
plt.ylabel(r'$\bar{\frac{||M_{20}(\delta)||^2}{LIM_{20}}}$')
plt.show()



# plt.plot(np.arange(-1,5,0.01), np.sqrt((-1 + 20**2 - 20*np.arange(-1,5,0.01))/(-1 + 20)))
sensor_range = range(2, 11, 2)
# Create a 2-row subplot: scatter plot in the first row and histograms in the second row
fig, ax = plt.subplots(1, 1, figsize=(12, 12))#, gridspec_kw={'width_ratios': [1, 3], 'height_ratios': [3, 1]})

# Scatter plot (center part)
for i, n in enumerate(sensor_range):
    x = np.array(sum_corrs_by_n[n])
    y = np.array(T2_ratios_by_n[n])

    # Filter out NaN and Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Scatter plot for each sensor group
    ax.scatter(x_clean, y_clean, alpha=0.5, edgecolors='k', 
                      color=color_list[i % len(color_list)], label=f'p = {n}')
    

    
for i, n in enumerate(sensor_range):
    x = np.array(sum_corrs_by_n[n])
    y = np.array(T2_ratios_by_n[n])

    # Filter out NaN and Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]


    
    ax.scatter(np.mean(x_clean), np.mean(y_clean), marker = 's',s=500, alpha=1, edgecolors='k', 
                      color=color_list[i % len(color_list)])
plt.xlabel(r'$\mathcal{L}$ [-]')
plt.ylabel(r'$\frac{||M_{20}(\delta)||^2}{LIM_{20}}$')
ax.legend(title="Number of sensors")
ax.grid(True)







