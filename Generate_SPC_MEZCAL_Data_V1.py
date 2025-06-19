# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:35:18 2024

@author: tbeene
"""
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal
import random
from collections import Counter
import math 
from scipy.special import multigammaln
from numpy.linalg import slogdet, det
import warnings
warnings.filterwarnings('ignore')
from collections import deque
from scipy.linalg import qr, solve_triangular

def label_nodes_by_cluster_list(mst_edges, num_nodes):
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from([(u, v) for u, v, _ in mst_edges])

    clusters = list(nx.connected_components(G))
    labels = [None] * num_nodes
    
 

    for cluster_id, component in enumerate(clusters):
        for node in component:
            labels[node] = cluster_id

    return labels

def generate_multivariate_data(data, run_length):
    """
    Generate synthetic multivariate random data using the mean and covariance of the original data.
    
    Parameters:
    - data: pd.DataFrame, original multivariate data
    - run_length: int, number of synthetic samples to generate
    
    Returns:
    - pd.DataFrame with synthetic data
    """
    means = np.mean(data, axis=0)
    cov = np.cov(data.T)
    synthetic_data = multivariate_normal(mean=means, cov=cov, size=run_length)
    return pd.DataFrame(synthetic_data, columns=data.columns)




def score_found_grouping(labels, data):
    
    """
    Compute the summed internal energy (negative log-likelihood) for each cluster.

    Parameters:
    - labels: array-like, cluster assignments (one per sample)
    - corr_matrix: (n_samples x n_samples) full correlation matrix
    - data: pandas DataFrame or NumPy array, shape (n_samples, n_features)

    Returns:
    - F: float, total internal energy across clusters
    """
    
    F = 0.0
    Lambda = 0.25

    # Convert pandas DataFrame to NumPy array if needed
    if hasattr(data, 'values'):
        data = data.T.values

    labels = np.array(labels)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue

        cluster_data = data[indices, :].T # shape: (n_cluster, n_features)
        # Sum evidence of all models
        entropy = calculate_evidence(cluster_data) 
  
           
        F += entropy

    return F



def calculate_evidence(X, N0=0):
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
    S = np.cov(X[:-1, :].T, bias=True)

    # Prior covariance: identity matrix + small noise
    Sigma_prior = np.eye(d) / d + np.random.rand(d, d) * 1e-5

    # Shrinkage intensity
    lambda_ = N0 / (N0 + n)

    # MAP covariance estimate
    Sigma_map = lambda_ * Sigma_prior + (1 - lambda_) * S
 
    # Use slogdet for numerical stability
    det1 = max(10e-10,det(Sigma_map))
    logdet = np.log(det1)

    # Compute log entropy of the Gaussian
    log_entropy = -0.5 * (d * np.log(2 * np.pi) + logdet)

    # Return average log marginal likelihood per dimension
    return -log_entropy 





def simulated_annealing_prune(mst_edges, num_nodes, all_edges, corr_matrix, data,
                               initial_temp=1, cooling_rate=0.99,
                               min_temp=1e-10, max_iter=500):
    
 
    def objective_function(edges, T):
        labels = label_nodes_by_cluster_list(edges, num_nodes)

        
        F = score_found_grouping(labels, data) #internal_energy(edges, labels)
        return F

    def get_neighbor(edges, all_edges):
        # Try removing one edge and make sure the graph remains valid
        for _ in range(20): # Try up to 10 times to find a valid neighbor
            temp_edges = edges.copy()
            edge_to_remove = random.choice(edges)
            
            # Find all edges with weight larger then removed weight
            filtered_edges = sorted(
                [e for e in all_edges if e not in temp_edges and e[2] > edge_to_remove[2]],
                key=lambda x: x[2]
            )
            
            
            if filtered_edges != []:
                edge_to_add = random.choice(filtered_edges)
                temp_edges.append(edge_to_add)


            temp_edges.remove(edge_to_remove)
            
            F_list.append(best_score)
            # print(best_score)

            
            

            # Check if degrees are valid (or use full connectivity check)
            degree = [0] * num_nodes
            for x, y, _ in temp_edges:
                degree[x] += 1
                degree[y] += 1
            if all(d > 0 for d in degree):
                
                return temp_edges
        return edges # fallback: return original if no good neighbor found

    sorted_edges = sorted(mst_edges, key=lambda x: -x[2])
    current_edges = mst_edges.copy()
    best_edges = current_edges.copy()

    T = initial_temp
    iteration = 0
    
    current_score= objective_function(current_edges, T)
    best_score = current_score
    
    

    entropy_list = []
    U_list = []
    F_list = []
    time_list = []
    
    F_list.append(best_score)
  

    while iteration < max_iter:
        neighbor_edges = get_neighbor(current_edges, all_edges)
        neighbor_score = objective_function(neighbor_edges, T)
        
   
        

        delta = neighbor_score - current_score
        

        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / T):
            
            current_edges = neighbor_edges
            current_score = neighbor_score

            labels = label_nodes_by_cluster_list(current_edges, num_nodes)
           
      

            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges.copy()
                best_T = T
                
                

        T *= cooling_rate
        iteration += 1

   
    return best_edges, F_list


def generate_balanced_list(cluster_numbers, total_length):
    if total_length % cluster_numbers != 0:
        raise ValueError("The total_length must be divisible by cluster_numbers for equal distribution.")
    
    counts_per_cluster = total_length // cluster_numbers
    numbers = []
    
    # Create an evenly distributed list
    for i in range(0, cluster_numbers):
        numbers.extend([i] * counts_per_cluster)
    
    # Shuffle the list to randomize it
    random.shuffle(numbers)
    
    # Validate the result
    if len(numbers) == total_length and all(count == counts_per_cluster for count in Counter(numbers).values()):
        return numbers
    else:
        raise ValueError("Failed to generate a balanced list.")
        

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




def max_spanning_tree_edges(corr_matrix, labels=None):
    """
    Computes the Maximum Spanning Tree using Kruskal's algorithm on a correlation matrix.

    Parameters:
    - corr_matrix: 2D numpy array representing the correlation matrix.
    - labels: Optional list of labels for the nodes. If None, numerical indices are used.

    Returns:
    - List of tuples representing edges in the Maximum Spanning Tree.
    """
    n = corr_matrix.shape[0]
    if labels is None:
        labels = list(range(n))

    # Create a list of all edges with their weights
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            weight = corr_matrix[i, j]
            edges.append((weight, labels[i], labels[j]))

    # Sort edges in decreasing order of weight
    edges.sort(reverse=True)

    # Kruskal's algorithm - Union Find setup
    parent = {label: label for label in labels}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]] # Path compression
            u = parent[u]
        return u

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u
            return True
        return False

    mst_edges = []
    all_edges = []
    for weight, u, v in edges:
        all_edges.append((u, v, weight))
        if union(u, v):
            mst_edges.append((u, v, weight))
            if len(mst_edges) == n - 1:
                break

    return mst_edges, all_edges




def label_nodes_by_cluster_list(mst_edges, num_nodes):
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from([(u, v) for u, v, _ in mst_edges])

    clusters = list(nx.connected_components(G))
    labels = [None] * num_nodes
    
 

    for cluster_id, component in enumerate(clusters):
        for node in component:
            labels[node] = cluster_id

    return labels


    
def apply_DBSCAN(df2, data_real, T_list, cluster_matrix= None):
    
    
    n = df2.shape[0]
    k = df2.shape[1]
    
    
    


    corr_matrix = df2.copy().corr().replace(np.nan, 0) 
    N0 = 0.1
    
    Corr_0 = np.identity(k) + 10e-6 * np.random.rand(k,k)
    distance_matrix = corr_matrix.values  #Corr_0*(N0/(n-N0)) + (1-N0/(n-N0)) * corr_matrix.values 

    mst, all_edges = max_spanning_tree_edges(np.abs(distance_matrix))


    pruned_mst, F_list =  simulated_annealing_prune(mst, len(distance_matrix), all_edges, distance_matrix, df2.copy())
    

    
    
  
    labelsDBSCAN = label_nodes_by_cluster_list(pruned_mst, k)

 

    df2 = df2.T
    cluster_dict_filter = {}
    noise_clusters = {}
    noise_clusters[-1] = []
    

    for jj in range(0, np.max(labelsDBSCAN) + 1, 1):
        cluster_dict_filter[jj] = []
 
            
    ii = 0
    for key, value in df2.iterrows():
        
        if labelsDBSCAN[ii] != -1:
            cluster = labelsDBSCAN[ii]
            cluster_dict_filter[cluster].append(value)
        else:
            cluster = labelsDBSCAN[ii]
            noise_clusters[cluster].append(value)
            
        ii += 1
        
    

    # Convert cluster data to pandas df
    for cluster, value in cluster_dict_filter.items():
        cluster_df = pd.DataFrame(value)
        cluster_dict_filter[cluster] = cluster_df
        
    for cluster, value in noise_clusters.items():
        cluster_df = pd.DataFrame(value)
        noise_clusters[cluster] = cluster_df
        
  
    
    # Calculate cluster statistics
    key_max = 0
    SSMEWMA_Cluster_Sum = 0
    count_clusters = 0
    
    for cluster_key, cluster in cluster_dict_filter.copy().items():

        cluster_matrix[len(cluster)][n] += len(cluster)
        SSMEWMA_Cluster_Sum += len(cluster)
                
    # Fill noise cluster in matrix
    # if cluster_matrix != []:
    cluster_matrix[1][n] += 96 - SSMEWMA_Cluster_Sum
    
    

    

    T_list.append((-F_list[-1]))

        
    return cluster_dict_filter, noise_clusters, cluster_matrix, T_list

def apply_fixed_clustering(data, cluster_size, data_real):
    """This function returns a list of random predefined clusters based on the cluster size input"""
    
    df_real = pd.DataFrame(data_real)
    df2 = pd.DataFrame.from_dict(data)
    keys = df2.T.keys()

    cluster_numbers = len(keys)//cluster_size
    
    # Generate random labels
    labels = generate_balanced_list(cluster_numbers, len(keys))

    
    # Generate dictionary with clusters
    ii = 0
    cluster_dict = {}
    

    for jj in range(0, np.max(labels) + 1, 1):
        cluster_dict[jj] = []
 
    for key, value in df2.iterrows():
        cluster = labels[ii]
        cluster_dict[cluster].append(value)
        ii += 1
        
    # Convert cluster data to pandas df
    for cluster, value in cluster_dict.items():
        cluster_df = pd.DataFrame(value)
        cluster_dict[cluster] = cluster_df
        
        
    return cluster_dict



    
    

        
    

    