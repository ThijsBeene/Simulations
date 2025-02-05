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

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]




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
        
def mds_embedding(distance_matrix, n_components=2):
    """Convert a distance matrix into a coordinate matrix using MDS."""
    from sklearn.manifold import MDS
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    return coords

def adjust_clusters_to_equal_size(labels, n_clusters, n_points):
    """Adjust cluster labels to ensure each cluster has equal size."""
    from collections import Counter
    counts = Counter(labels)
    target_size = n_points // n_clusters
    cluster_indices = {i: np.where(labels == i)[0] for i in range(n_clusters)}

    for i in range(n_clusters):
        while counts[i] > target_size:
            for j in range(n_clusters):
                if counts[j] < target_size:
                    move_idx = cluster_indices[i][0]
                    labels[move_idx] = j
                    counts[i] -= 1
                    counts[j] += 1
                    cluster_indices[i] = cluster_indices[i][1:]
                    cluster_indices[j] = np.append(cluster_indices[j], move_idx)
                    break

    return labels




def Load_Data(L):
    # Load actual MEZCAL data
    xls = pd.ExcelFile(r"C:\Users\tbeene\Desktop\Data\MEZCAL\Copy of BB3C9600.xlsx")
    df1 = pd.read_excel(xls, 'Data', skiprows = 10)
    
    
    
    Data_Dict_Stack = {}

    
    for index, row in df1.iterrows():
        if 4 < index < 15 or 18 < index < 25 or index == 34 or 47 < index < 56 or 66 < index < 76 or 80 < index < 104 or 104 < index < 116 or 116 < index < 123 or 124 < index < 127 or 128 < index < 138 or 138 < index < 149 or 149 < index < 168 or 169 < index < 179 or 180 < index < 197 or 199 < index < 249:
            if len(row[28:].dropna()) != 0 and isinstance(row[30:].dropna()[0], str) == False:

                Axis_Text = str(row['Test description'])
                Data_Dict_Stack[Axis_Text] = row[30:(30+L+1)].dropna().tolist()
    
        
    df2 = pd.DataFrame.from_dict(Data_Dict_Stack)
    
    # Drop all columns where the std is 0!
    # Some columns only contain the same value
    # df2 = df2.loc[:, df2.std() != 0]
    
    
    
    corr_matrix = df2.corr()
    
    # df2=(df2-df2.mean())/df2.std()

    return df2, corr_matrix


def generate_multivariate_data(cov, data, run_length):

    columns = data.columns
    
    means = np.mean(data, axis = 0)

    
    transformed_samples = multivariate_normal(mean = means, cov = cov, size = run_length).T

    new_matrix = pd.DataFrame()
    index = 0
    for column in columns:
        new_matrix[column] = transformed_samples[index]
        index += 1


    
    return new_matrix.T

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
 
    
    # for ii in range(len(column_headers)):
    #     if labelsDBSCAN[ii] != -1:
    #         cluster = labelsDBSCAN[ii]
    #         cluster_dict_filter[cluster].append(df2.T[column_headers[ii]])
    #     else:
    #         cluster = labelsDBSCAN[ii]
    #         noise_clusters[cluster].append(df2.T[column_headers[ii]])
            
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

    
    # Apply KMeans
    # df2 = pd.DataFrame.from_dict(data)
    # keys = df2.keys()
    # corr_matrix = df_real.corr()
    # dissimilarity = 1 - np.abs(corr_matrix)  
    # distance_matrix = dissimilarity
    # coords = mds_embedding(distance_matrix)
    # n_points = distance_matrix.shape[0]
    # n_clusters = cluster_numbers
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # labels = kmeans.fit_predict(coords)

    # # Adjust clusters to have equal sizes
    # labels = adjust_clusters_to_equal_size(labels, n_clusters, n_points)

    

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

def write_to_csv(data, filepath, filename):
    data.to_csv(filepath + '/' + filename, sep=',', index=False, encoding='utf-8')


def Get_Random_Clustered_MEZCAL_Data(data_real, cov, run_length):
    filepath = r"C:\Users\tbeene\Desktop\Data\Workcenter_Delft_AVL_OLnD_SPC_Data"
    filename = r"data.csv"
    
   
    # Generate data based on MEZCAL data
    #data_real, cov = Load_Data(32)
    
 
    data_random = generate_multivariate_data(cov, data_real, run_length)
    

    # clustered_data_COMBINED, cluster_data_COMBINED_noise, noise_cluster_number, total_cluster, biggest_cluster = apply_DBSCAN(data_random.T, DBSCAN_threshold, data_real.T)
    # clustered_data_SSEWMA = apply_fixed_clustering(data_random, cluster_size, data_real)
    
    
    # write_to_csv(data_random, filepath, filename)
    # write_to_csv(clustered_data_SSEWMA, filepath, filename)
    # write_to_csv(clustered_data_COMBINED, filepath, filename)
    # write_to_csv(cluster_data_COMBINED_noise, filepath, filename)


    return data_random #, clustered_data_SSEWMA, clustered_data_COMBINED, cluster_data_COMBINED_noise, noise_cluster_number, total_cluster, biggest_cluster



    
    

        
    

    