# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:35:18 2024

@author: tbeene
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from numpy.random import multivariate_normal
import random
from collections import Counter


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


def generate_multivariate_data(cov, data, run_length):
    """Generate synthetic multivariate random data using a given covariance matrix."""
    means = np.mean(data, axis=0)
    return pd.DataFrame(multivariate_normal(mean=means, cov=cov, size=run_length), columns=data.columns).T


def apply_DBSCAN(data, threshold, data_real):
    
    

    df2 = pd.DataFrame.from_dict(data)
    df2 = df2.T.sort_values(by=0,ascending=False)



    T = 1-threshold # If correlation between sensors is greater then threshold, they can be placed in same cluster
    corr_matrix = np.abs(df2.T.corr()) #.T.iloc[:,:-1].T
    
    
    # Invert covariance matrix to obtain a measure of distance
    dissimilarity = 1 - corr_matrix 
    dissimilarity = np.abs(dissimilarity)
    
    
    clustering = DBSCAN(eps=T, min_samples= 2, metric='precomputed')
    labelsDBSCAN = clustering.fit_predict(dissimilarity)
    
    column_headers = list(dissimilarity.columns.values)
    
 
    # Get DBSCAN and noise
    
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
    
    count_clusters = 0
    for key, cluster in cluster_dict_filter.items():
        count_clusters += 1
        
        if len(cluster) >= key_max:
            key_max = len(cluster)

    noise_cluster_number = len(noise_clusters[-1])
    total_cluster = count_clusters
    biggest_cluster = (96-noise_cluster_number)/count_clusters#key_max

        
    return cluster_dict_filter, noise_clusters, noise_cluster_number, total_cluster, biggest_cluster

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



    
    

        
    

    