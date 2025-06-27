import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal


def Load_Data(L):
    """Load and process MEZCAL data."""
    xls = pd.ExcelFile(r"C:\Users\tbeene\Desktop\Data\MEZCAL\Copy of BB3C9600.xlsx")
    df1 = pd.read_excel(xls, 'Data', skiprows=10)
    Data_Dict_Stack = {}

    valid_indices = set(range(5, 15)) | set(range(19, 25)) | {34} | set(range(48, 56)) | \
                    set(range(67, 76)) | set(range(81, 104)) | set(range(105, 116)) | \
                    set(range(117, 123)) | set(range(125, 127)) | set(range(129, 138)) | \
                    set(range(139, 149)) | set(range(150, 168)) | set(range(170, 179)) | \
                    set(range(181, 197)) | set(range(200, 249))

    for index, row in df1.iterrows():
        if index in valid_indices:
            if len(row[28:].dropna()) > 0 and not isinstance(row[30:].dropna()[0], str):
                Axis_Text = str(row['Test description'])
                Data_Dict_Stack[Axis_Text] = row[30:(30 + L + 1)].dropna().tolist()

    df2 = pd.DataFrame.from_dict(Data_Dict_Stack)
    corr_matrix = df2.corr()
    return df2, corr_matrix


def generate_multivariate_data(cov, data, run_length):
    """Generate synthetic multivariate random data using a given covariance matrix."""
    means = np.mean(data, axis=0)
    return pd.DataFrame(multivariate_normal(mean=means, cov=cov, size=run_length), columns=data.columns).T



def compute_Wnk(df, k):
    """Compute W_n_k as per the control chart method."""
    
    W1, W2, W3 = 0, 0, 0
    
    # Use numpy arrays to speed up calculations drastically!!
    data = df.values
    n = data.shape[1]
   

    
    for i in range(k):
        for j in range(k):
            if i != j:
                W1 += data[:, i].T @ data[:, j]

    for i in range(k, n):
        for j in range(k, n):
            if i != j:
                W2 += data[:, i].T @ data[:, j]

    for i in range(k):
        for j in range(k, n):
            W3 += data[:, i].T @ data[:, j]
            
            
            

    w_n_k = (W1 / (k * (k - 1)) + W2 / ((n - k) * (n - k - 1)) - 2 * W3 / (k * (n - k)))
    
    return w_n_k



def compute_s2_w(df, k):
    X = df.T.values


    """
    Calculates the variance of Wnk (biased estimator) for a given dataset X and change-point k.

    Parameters:
        X: numpy.ndarray
            The dataset of shape (n, p) where n is the number of samples and p is the number of features.
        k: int
            The change-point index (1 <= k < n).

    Returns:
        float
            The variance of Wnk.
    """
    n, p = X.shape
    
    if k <= 1 or k >= n - 1:
        raise ValueError("k must be between 2 and n-2 for variance estimation.")

    # Sample covariance matrix (assumed the same for pre- and post-change)
    Sigma0 = np.cov(X[:k, :], rowvar=False, bias=False)
    Sigma1 = np.cov(X[k:, :], rowvar=False, bias=False)
    
    # Compute trace of Sigma squared
    trace_Sigma1 = np.trace(Sigma0 @ Sigma0)
    trace_Sigma2 = np.trace(Sigma1 @ Sigma1)
    trace_Sigma3 = np.trace(Sigma0 @ Sigma1)

    term1 = (2 * trace_Sigma1) / (k * (k - 1))
    term2 = (2 * trace_Sigma2) / ((n - k) * (n - k - 1))
    term3 = (4 * trace_Sigma3) / (k * (n - k))

    variance = term1 + term2 + term3

    return variance





def Calculate_Change_Point(data, h_n_p):
    """
    Calculates change-point statistics with a dynamically determined threshold h_n_p.
    Optimized loop and guards.
    """
    Z_list = []
    lim_list = [0,0,0]

    # Limit found numerically
    lim_list.extend(2.3 + 0.59*np.log(np.arange(1,len(data.T)-2, 1)))
    

    for n in range(1, len(data.columns)+1): # Start at 4 to ensure valid k ranges

        data_subset = data.iloc[:, :n]

        W_n_k = []
        S_2_w = []

        # k ranges from 2 to n - 1 (inclusive)
        for k in range(2, n - 1):
            Wnk_val = compute_Wnk(data_subset, k)
            s2w_val = compute_s2_w(data_subset, k)

            W_n_k.append(Wnk_val)
            S_2_w.append(s2w_val)

        # Filter out invalid divisions and take max Z-stat
        filtered_data = [
            v / np.sqrt(s) if s > 0 and not np.isnan(v / s) and not np.isinf(v / s) else 0
            for v, s in zip(W_n_k, S_2_w)
        ]

        max_Z = max(filtered_data) if filtered_data else 0
        Z_list.append(max_Z)

    return Z_list, lim_list




def Calculate_Multivariate_Change_Point(full_data, lim = 3.2): 
    limit = lim
    z_statistics, lim_list = Calculate_Change_Point(full_data, limit)

    

    
    # plt.figure()
    # plt.style.use('science')
    # plt.rc('font', size=14)
    # plt.rc('axes', titlesize=18)
    # plt.rc('axes', labelsize=18)
    # plt.rc('xtick', labelsize=18)
    # plt.rc('ytick', labelsize=18)
    # plt.rc('legend', fontsize=18)
    # plt.grid()
    # plt.plot(z_statistics, label = '$Z_{max,n}$')
    # plt.plot(lim_list, label = 'Limit ($h$)', c = 'orange')
    # plt.xlabel('n [-]')
    # plt.ylabel('$Z_{max,n}$')
    # plt.legend()
    

    state = ["IC" if z <= lim else "OOC" for z, lim in zip(z_statistics, lim_list)]

    result_df = pd.DataFrame({
        "Z": z_statistics,
        "Limit": lim_list,
        "state": state
    })

    return result_df

#0.173426415569660434
def simulate_and_plot_change_detection(n, shift, lim): 

    
    data, cov = Load_Data(32) # Ensure Load_Data uses a matrix of observations as input
    data_random = generate_multivariate_data(cov, data, n)
    
    initial_data = data_random.T #apply_mean_shift(generate_random_data(), shift) #data_random.T# generate_random_data()#apply_mean_shift(generate_random_data(), shift)
   
    # import random
    # l = random.randint(1,95)
    # res=data_random.T.columns[l]
    # full_observation_vector = initial_data[res][:30]
    # initial_data[res][20:] += 5*np.std(full_observation_vector)


    full_data = initial_data.T
    
    h = 1 + 0.59*np.log(n)
   
    result = Calculate_Multivariate_Change_Point(full_data, h) # p =10: , p = 96: 2.9729964510579907

    
    

    return result, initial_data

def determine_h_n_p():
    filepath = r"C:\Users\tbeene\Desktop\Simulation\Numerical_ARL_IC_Calculations\MCPD_SHORT_RANGE"
    filename_final = f"simulation_output_results.csv"
    
    Z_max_n_values = []
    t1 = time.time()
    runs = 50
    N = 30
    
    h_list = []
    
    for n in range(N, N+1):
     
        for ii in range(runs):

            print(ii)

            result,initial_data = simulate_and_plot_change_detection(n, 0, 1)
            Z_max_n_values.extend([i for i in result['Z'] if i != 0])
            
        
            
        false_alarm_rate = (1/25)
        
        def simulate_ssmewma(h, T2_values, ARL_IC_Overall):
            ARL_STD = []
            first_OOC_list = []
            for val in T2_values:
                xx = 0
                for idx,ob in enumerate(val):
                    if ob > h:
                        first_OOC_list.append(idx)
                        xx = 1
                        break
                if xx == 0:
                    first_OOC_list.append(np.inf)

            ARL_estimated = np.mean(first_OOC_list)

            return ARL_estimated - ARL_IC_Overall

        
        
        h_n_p = np.quantile(Z_max_n_values, 1-false_alarm_rate)
        h_list.append(h_n_p)
        print(f'h =',h_n_p)
    return h_list, Z_max_n_values
  

# Function to generate multivariate random data
def generate_random_data(process_readings=10, observations=40, mean=0, std=1):
    data = np.random.normal(loc=mean, scale=std, size=(observations, process_readings))
    return pd.DataFrame(data, columns=[f'Process_{i+1}' for i in range(process_readings)])

# Function to apply mean shift
def apply_mean_shift(data, shift, percentage=1):
    shifted_data = data.copy()
    num_processes_to_shift = int(data.shape[1] * percentage)
    processes_to_shift = np.random.choice(data.columns, num_processes_to_shift, replace=False)
    shifted_data.loc[4:, processes_to_shift] += shift
   

    
    return shifted_data


# # # # # h_n_p limit calculation
# h_n_p, Z_max_n_values = determine_h_n_p()  
# # # # # # # plt.hist(Z_max_n_values, bins = 100)
# print(h_n_p)
# # # plt.style.use('science')


# plt.figure()
# plt.title('$h_{n,p}$ calculation for n varying for 96 process readings')
# plt.ylabel('$h_{n,p}$ [-]')
# plt.xlabel('Observations [-]')
# plt.grid()
# plt.plot(h_n_p)
# plt.plot(1 + 0.59*np.log(np.arange(5, 30, 1)))
# plt.axvline(h_n_p, label = 'Quantile corresponding to $ARL_{IC}$ = 25', c = 'orange', linestyle = '--')
# plt.hist(Z_max_n_values, bins = 100)
# plt.xlabel('$Z_{max,n}$ [-]')
# plt.ylabel('Counts [-]')
# plt.grid()
# plt.legend()



# # # # Verification with paper table 1
# delta = [0.5,1.0,1.5,2,2.5,3]
# # # ARL_Actual_kwart = [10.4,5.6,4.3,3.3,3.2]
# # # ARL_Actual_half = [5.8,4.1,3.3,3.1,2.9]
# # # ARL_Actual_driekwart = [10.6,4,3.5,3.2,2.9,2.8]


# # # ARL_Tau_5 = [24.6,4.2,3.3,3.1,2.9]
# # # ARL_Tau_5_75 = [15.3,8.3,4.2,3.2,2.9,3.0]

# ARL_Tau_Shift_1_Tau_5 = [9.6, 6.6, 3.7, 3, 3, 3]
# # # ARL_Tau_Shift_1_Tau_15 = [8.4, 4, 3, 2.9, 2.9, 2.6]

# ARL = []
# ARL_ERR = []
# for shift in delta:
#     OOC = []
#     for ii in range(50):
#         result, initial_data = simulate_and_plot_change_detection(10, shift,  40)
#         print(ii)
#         count = 0
#         for ii in range(4, 40):
#             if result['state'][ii] == 'OOC':
#                 # Shift starts at 4
#                 count = ii - 4 + 1
#                 OOC.append(count)
#                 print(count)
#                 break
                
#     print(OOC)
#     ARL.append(np.mean(OOC))
#     # SEM of 95% CI
#     ARL_ERR.append(1.96*2*np.std(OOC)/np.sqrt(len(OOC)))
    
    

# plt.style.use('science')
# plt.figure()

# plt.errorbar(delta, ARL_Tau_Shift_1_Tau_5, fmt="r--o", label="Results from Li et al")
# plt.errorbar(delta, ARL, yerr=ARL_ERR, fmt="b--o", label="Implementation results")
# plt.ylabel('ARL')
# plt.xlabel('$\delta$')
# plt.title('Mean shift VS ARL')
# plt.grid()
# plt.legend()
# plt.show()

