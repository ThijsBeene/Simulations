import time
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import math

def write_to_csv(data, filepath, filename):
    data.to_csv(filepath + '/' + filename, sep=',', index=True, encoding='utf-8')

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
    return pd.DataFrame(multivariate_normal(mean=means, cov=cov, size=run_length), columns=data.columns)


def find_W_n_k(df, k):
    """Compute W_n_k as per the control chart method."""
    n = df.shape[1]
    W1, W2, W3 = 0, 0, 0
    
    # Use numpy arrays to speed up calculations drastically!!
    data = df.values

    for i in range(0, k):
        for j in range(0, k):
            if i != j:
                W1 += data[:, i].T @ data[:, j]

    for i in range(k, n):
        for j in range(k, n):
            if i != j:
                W2 += data[:, i].T @ data[:, j]

    for i in range(0, k):
        for j in range(k, n):
            W3 += data[:, i].T @ data[:, j]

    w_n_k = (W1 / (k * (k - 1)) + W2 / ((n - k) * (n - k - 1)) - 2 * W3 / (k * (n - k)))
    return w_n_k


def find_s_2_w(df, k):
    """Compute s^2_w based on variance estimation."""
    n = df.shape[1]
    S1, S2, S3 = 0.0, 0.0, 0.0

    subset1 = df.iloc[:, :k].values
    subset2 = df.iloc[:, k:n].values
    
    
    for j in range(k):
        for l in range(k):
            if j != l:
                mean_vector = np.mean(np.delete(subset1, [j, l], axis=1), axis=1, keepdims=True)
                delta_j = subset1[:, j] - mean_vector
                delta_l = subset1[:, l] - mean_vector
                S1 += (delta_j * delta_j.T * delta_l * delta_l.T)
    
    for j in range(n-k):
        for l in range(n-k):
            if j != l:
                mean_vector = np.mean(np.delete(subset2, [j, l], axis=1), axis=1, keepdims=True)
                delta_j = subset2[:, j] - mean_vector
                delta_l = subset2[:, l] - mean_vector
                S2 += (delta_j * delta_j.T * delta_l * delta_l.T)
    
    # Index 0 is k because we sliced the df from k to n!!!
    for j in range(k):
        for l in range(n-k):
            mean_vector_0 = np.mean(np.delete(subset1, [j], axis=1), axis=1, keepdims=True)
            mean_vector_1 = np.mean(np.delete(subset2, [0], axis=1), axis=1, keepdims=True)
            delta_j = subset1[:, j] - mean_vector_0
            delta_l = subset2[:, 0] - mean_vector_1
            S3 += (delta_j * delta_j.T * delta_l * delta_l.T)
    
    S1 = np.trace((1/k*(k-1))*S1)
    S2 = np.trace((1/((n-k)*(n-k-1)))*S2)
    S3 = np.trace((1/((k)*(n-k)))*S3)
    
    s2_w = (2 / (k * (k - 1))) * S1 + (2 / ((n - k) * (n - k - 1))) * S2 + (4 / (k * (n - k))) * S3
    
    return s2_w




# def find_s_2_w(df, k):
#     """Compute s^2_w based on variance estimation with optimization."""
#     n = df.shape[1]
#     subset1 = df.iloc[:, :k].values
#     subset2 = df.iloc[:, k:].values

#     # Compute S1
#     mean_subset1 = np.mean(subset1, axis=1, keepdims=True)
#     deltas1 = subset1 - mean_subset1
#     S1 = np.sum(np.triu(np.dot(deltas1, deltas1.T)**2, k=1)) / (k * (k - 1))

#     # Compute S2
#     mean_subset2 = np.mean(subset2, axis=1, keepdims=True)
#     deltas2 = subset2 - mean_subset2
#     S2 = np.sum(np.triu(np.dot(deltas2, deltas2.T)**2, k=1)) / ((n - k) * (n - k - 1))

#     # Compute S3
#     mean_subset1_only = np.mean(subset1, axis=1, keepdims=True)
#     mean_subset2_only = np.mean(subset2, axis=1, keepdims=True)
#     S3 = np.sum(
#         (subset1 - mean_subset1_only)[:, :, None] * (subset2 - mean_subset2_only)[:, None, :]**2
#     ) / (k * (n - k))

#     # Combine to calculate s2_w
#     s2_w = (2 / (k * (k - 1))) * S1 + (2 / ((n - k) * (n - k - 1))) * S2 + (4 / (k * (n - k))) * S3

#     return s2_w

def Calculate_Change_Point(data, limit):
    """Calculate change-point statistics."""
    Z_list = []
    lim_list = [limit] * (len(data.columns)+1)

    for n in range(0, len(data.columns) + 1):
        
        W_n_k = []
        S_2_w = []
        current_matrix = data.iloc[:, :n]

        for k in range(2, n - 1):
            W_n_k.append(find_W_n_k(current_matrix, k))
            S_2_w.append(find_s_2_w(current_matrix, k))

        # Should we take the square root of the variance to get the standard deviation?
        # Also filter out nan and inf
        filtered_data = [v / s if not math.isnan(v / s) and not math.isinf(v / s) else 0
                         for v, s in zip(W_n_k, S_2_w)]

        Z_list.append(max(filtered_data) if filtered_data else 0)

    return Z_list, lim_list


def Calculate_Multivariate_Change_Point(full_data, lim = 0.2417572721771737):
    limit = lim
    z_statistics, lim_list = Calculate_Change_Point(full_data, limit)
    

    state = ["IC" if z <= lim else "OOC" for z, lim in zip(z_statistics, lim_list)]

    result_df = pd.DataFrame({
        "Z": z_statistics,
        "Limit": lim_list,
        "state": state
    })

    return result_df


def simulate_and_plot_change_detection(run, n, shift, lim = 0.3697572721771737): # Lim 100 = 1.2514957122952044, lim 200 = 1.6157644052891473
    #1.3549676671757231
    
    data, cov = Load_Data(32) # Ensure Load_Data uses a matrix of observations as input
    data_random = generate_multivariate_data(cov, data, n)
    
    

    initial_data = apply_mean_shift(generate_random_data(), shift)
   
    # import random
    # l = random.randint(1,96)
    # res=data_random.columns[l]
    # full_observation_vector = initial_data[res][:20]
    # initial_data[res][20:] += 7*np.std(full_observation_vector)
        
    
    full_data = initial_data.T
    
    result = Calculate_Multivariate_Change_Point(full_data, lim)
    
    filename_data = f"run_DATA_{run}_{np.round(lim,2)}.csv"
    filename_MCPD = f"run_MCPD_{run}_{np.round(lim,2)}.csv"
    
    # write_to_csv(full_data, filepath, filename_data)
    # write_to_csv(result, filepath, filename_MCPD)
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(result.index, result["Z"], label='Z Statistic', color='blue', marker='o')
    # plt.axhline(y=lim, color='red', linestyle='--', label='Control Limit (0.04)')
    # plt.title('Change Point Detection - Z Statistic vs Control Limit')
    # plt.xlabel('Observation Number')
    # plt.ylabel('Z Statistic')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    

    return result

def determine_h_n_p():
    filepath = r"C:\Users\tbeene\Desktop\Simulation\Numerical_ARL_IC_Calculations\MCPD_SHORT_RANGE"
    filename_final = f"simulation_output_results.csv"
    
    Z_max_n_values = []
    t1 = time.time()
    runs = 300
    
    for ii in range(runs):
        print(ii)
        p = 96
        n = 30
        result = simulate_and_plot_change_detection(8, n, n)
        Z_max_n_values.extend([i for i in result['Z'] if i != 0])
        
    
        
    false_alarm_rate = 1/100 # Corresponds to an ARL_IC of 100
    h_n_p = np.quantile(Z_max_n_values, 1-false_alarm_rate)
    return h_n_p, Z_max_n_values
  
# h_n_p, Z_max_n_values = determine_h_n_p()  
# print(h_n_p)
# plt.hist(Z_max_n_values)

# Function to generate multivariate random data
def generate_random_data(process_readings=10, observations=30, mean=0, std=1):
    """
    Generate multivariate random data for process readings and observations.

    Args:
        process_readings (int): Number of process readings.
        observations (int): Number of observations.
        mean (float): Mean of the random data.
        std (float): Standard deviation of the random data.

    Returns:
        pd.DataFrame: A DataFrame containing the random data.
    """
    data = np.random.normal(loc=mean, scale=std, size=(observations, process_readings))
    return pd.DataFrame(data, columns=[f'Process_{i+1}' for i in range(process_readings)])

# Function to apply mean shift
def apply_mean_shift(data, shift, percentage=0.25):
    """
    Apply a mean shift to a percentage of process readings.

    Args:
        data (pd.DataFrame): DataFrame containing the random data.
        shift (float): Value of the mean shift.
        percentage (float): Percentage of process readings to apply the mean shift.

    Returns:
        pd.DataFrame: A DataFrame with the mean shift applied.
    """
    shifted_data = data.copy()
    num_processes_to_shift = int(data.shape[1] * percentage)
    processes_to_shift = np.random.choice(data.columns, num_processes_to_shift, replace=False)
    shifted_data.loc[14:, processes_to_shift] += shift

    
    return shifted_data

# # Generate the random data
# random_data = generate_random_data()

# # Apply the mean shift
# shifted_data = apply_mean_shift(random_data)

# # Display the first few rows of the original and shifted data
# print("Original Data:")
# print(random_data.head())

# print("\nShifted Data:")
# print(shifted_data.head())

   
  
# h_n_p, Z_max_n_values = determine_h_n_p()  
# print(h_n_p)


# plt.axvline(h_n_p, label = 'Quantile corresponding to $ARL_{IC}$ = 100')
# plt.hist(Z_max_n_values, bins = 50)
# plt.xlabel('$Z_{max,n}$ [-]')
# plt.ylabel('Counts [-]')
# plt.legend()
# # p = 96
# n = 30

# delta = [1.5,2,2.5,3]
# ARL_Actual = [5.6,4.3,3.3,3.2]

# ARL = []
# ARL_ERR = []
# for shift in delta:
#     OOC = []
#     for ii in range(1000):
#         print(ii)
#         result = simulate_and_plot_change_detection(8, 30, shift)
#         count = 0
#         for ii in range(14, 30):
#             if result['state'][ii] == 'OOC' and count == 0:
#                 count = ii - 14 
#                 OOC.append(count)
#                 break
#     ARL.append(np.mean(OOC))
#     ARL_ERR.append(np.std(OOC))
    
# plt.scatter(delta, ARL_Actual, label = 'Paper results')
# plt.errorbar(delta, ARL, yerr = ARL_ERR, fmt="g--o", ecolor = "black", label = 'Simulated')
# plt.xlabel('$\delta$')
# plt.ylabel('ARL')
# plt.grid()
# plt.legend()
# print(np.mean(OOC))
# print(np.std(OOC))

# result = simulate_and_plot_change_detection(8, 30)    
# plt.plot(result['Z'])
# plt.plot(result['Limit'])


# def find_s_2_w(df, k):
#     """Compute s^2_w based on variance estimation with optimization."""
#     n = df.shape[1]
#     subset1 = df.iloc[:, :k].values
#     subset2 = df.iloc[:, k:].values

#     # Compute S1
#     mean_subset1 = np.mean(subset1, axis=1, keepdims=True)
#     deltas1 = subset1 - mean_subset1
#     S1 = np.sum(np.triu(np.dot(deltas1, deltas1.T)**2, k=1)) / (k * (k - 1))

#     # Compute S2
#     mean_subset2 = np.mean(subset2, axis=1, keepdims=True)
#     deltas2 = subset2 - mean_subset2
#     S2 = np.sum(np.triu(np.dot(deltas2, deltas2.T)**2, k=1)) / ((n - k) * (n - k - 1))

#     # Compute S3
#     mean_subset1_only = np.mean(subset1, axis=1, keepdims=True)
#     mean_subset2_only = np.mean(subset2, axis=1, keepdims=True)
#     S3 = np.sum(
#         (subset1 - mean_subset1_only)[:, :, None] * (subset2 - mean_subset2_only)[:, None, :]**2
#     ) / (k * (n - k))

#     # Combine to calculate s2_w
#     s2_w = (2 / (k * (k - 1))) * S1 + (2 / ((n - k) * (n - k - 1))) * S2 + (4 / (k * (n - k))) * S3

#     return s2_w
