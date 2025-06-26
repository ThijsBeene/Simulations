import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import multivariate_normal
import random
import math 
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import scienceplots
from scipy.special import multigammaln
from scipy.stats import invwishart
from numpy.linalg import slogdet, det
import warnings
warnings.filterwarnings('ignore')
import time
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
        F += calculate_evidence(cluster_data) 

  
    return (F)




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




def tabu_search_prune(mst_edges, num_nodes, all_edges, corr_matrix, data,
                       tabu_size=50, max_iter=50):
    
    t1 = time.time()
    def objective_function(edges):
        labels = label_nodes_by_cluster_list(edges, num_nodes)
        F = score_found_grouping(labels, data)
        return F

    def get_neighbor(current_edges, all_edges, tabu_list):
        candidate_neighbors = []

        # Try several valid neighbors and rank them
        for _ in range(20): # try 20 random edge swaps
            temp_edges = current_edges.copy()
            edge_to_remove = random.choice(current_edges)
            filtered_edges = sorted(
                [e for e in all_edges if e not in temp_edges and e[2] > edge_to_remove[2]],
                key=lambda x: x[2]
            )

            if not filtered_edges:
                continue

            edge_to_add = random.choice(filtered_edges)
            temp_edges.remove(edge_to_remove)
            temp_edges.append(edge_to_add)
            temp_edges_tuple = tuple(sorted(temp_edges))

            # Check graph validity and tabu status
            degree = [0] * num_nodes
            for x, y, _ in temp_edges:
                degree[x] += 1
                degree[y] += 1

            F_list.append(best_score)
            time_list.append(time.time() - t1)
            
            if all(d > 0 for d in degree) and temp_edges_tuple not in tabu_list:
                score = objective_function(temp_edges)
                
                candidate_neighbors.append((score, temp_edges, temp_edges_tuple))

        # Return the best valid non-tabu neighbor
        if candidate_neighbors:
            candidate_neighbors.sort(key=lambda x: x[0])
            return candidate_neighbors[0][1], candidate_neighbors[0][2], candidate_neighbors[0][0]
        else:
            return current_edges, tuple(sorted(current_edges)), objective_function(current_edges)

    current_edges = mst_edges.copy()
    best_edges = current_edges.copy()
    best_score = objective_function(current_edges)

    tabu_list = deque(maxlen=tabu_size)
    tabu_list.append(tuple(sorted(current_edges)))

    F_list = [best_score]
    time_list = [time.time() - t1]

    for iteration in range(max_iter):
        neighbor_edges, neighbor_signature, neighbor_score = get_neighbor(current_edges, all_edges, tabu_list)
        

        if neighbor_score < best_score:
           
            best_score = neighbor_score
            best_edges = neighbor_edges.copy()

        current_edges = neighbor_edges
        tabu_list.append(neighbor_signature)
        

    return best_edges, F_list, time_list





def greedy_edge_removal(mst_edges, num_nodes, all_edges, corr_matrix, data, max_iter=50, top_k_remove=50):
    t1 = time.time()
    
    def objective_function(edges):
        labels = label_nodes_by_cluster_list(edges, num_nodes)
        return score_found_grouping(labels, data)

    def is_valid_graph(edges):
        degree = [0] * num_nodes
        for x, y, _ in edges:
            degree[x] += 1
            degree[y] += 1
        return all(d > 0 for d in degree)

    current_edges = mst_edges.copy()
    best_edges = current_edges.copy()
    best_score = objective_function(current_edges)
    
    score_list = []
    time_list = []
    score_list.append(best_score)
    
    
    time_list.append(time.time() - t1)
    

    for _ in range(max_iter):
        # Only consider top-k highest weight edges for removal

        candidate_remove_edges = sorted(current_edges, key=lambda x: -x[2])
        candidate_add_edges = sorted([e for e in all_edges if e not in current_edges], key=lambda x: -x[2])[:20]
        
        #print(candidate_add_edges)
        #print(candidate_remove_edges)
        
        
        improved = False
        
        # Sampling:
        #for edge_to_remove, edge_to_add in zip(candidate_remove_edges, candidate_add_edges):
            
        for edge_to_remove in candidate_remove_edges:
            for edge_to_add in candidate_add_edges:
                temp_edges = current_edges.copy()
                temp_edges.remove(edge_to_remove)
                temp_edges.append(edge_to_add)
                
 
                score_list.append(best_score)
                time_list.append(time.time() - t1)

                if is_valid_graph(temp_edges):
                    temp_score = objective_function(temp_edges)
                    
                    
                    if temp_score < best_score:
          
                        
                        current_edges = temp_edges
                        best_edges = temp_edges
                        best_score = temp_score
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    return best_edges, score_list, time_list




def simulated_annealing_prune(mst_edges, num_nodes, all_edges, corr_matrix, data,
                               initial_temp=1.0, cooling_rate=0.99,
                               min_temp=1e-5, max_iter=1000):
    import time, random, math

    t1 = time.time()

    def objective_function(edges):
        labels = label_nodes_by_cluster_list(edges, num_nodes)
        return score_found_grouping(labels, data)  # Lower is better

    def get_neighbor(edges, all_edges):
        for _ in range(20):  # Try multiple times to find a valid neighbor
            temp_edges = edges.copy()
            edge_to_remove = random.choice(temp_edges)
            temp_edges.remove(edge_to_remove)

            # Find candidate edges not in current set and with higher weight
            candidate_edges = [e for e in all_edges if e not in temp_edges and e[2] > edge_to_remove[2]]
            if not candidate_edges:
                continue

            edge_to_add = random.choice(candidate_edges)
            temp_edges.append(edge_to_add)

            # Check if all nodes are still connected (basic degree check)
            degree = [0] * num_nodes
            for x, y, _ in temp_edges:
                degree[x] += 1
                degree[y] += 1
            if all(d > 0 for d in degree):
                return temp_edges

        return edges  # fallback

    current_edges = mst_edges.copy()
    best_edges = current_edges.copy()

    T = initial_temp
    iteration = 0

    current_score = objective_function(current_edges)
    best_score = current_score

    F_list = [best_score]
    time_list = [0.0]

    while iteration < max_iter and T > min_temp:
        neighbor_edges = get_neighbor(current_edges, all_edges)
        neighbor_score = objective_function(neighbor_edges)

        delta = neighbor_score - current_score

        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / T):
            current_edges = neighbor_edges
            current_score = neighbor_score

            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges.copy()

        F_list.append(best_score)
        time_list.append(time.time() - t1)

        T *= cooling_rate
        iteration += 1

    return best_edges, F_list, time_list





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



plt.style.use('science')

def visualize_mst(n, mst_edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, weight in mst_edges:
        G.add_edge(u, v, weight=round(weight, 2))
    pos = nx.spring_layout(G, seed=42)
    components = list(nx.connected_components(G))
    component_colors = np.linspace(0, 1, len(components))
    node_color_map = {}
    cmap = plt.get_cmap('tab20')
    for color, component in zip(component_colors, components):
        for node in component:
            node_color_map[node] = cmap(color)
    node_colors = [node_color_map[node] for node in G.nodes()]
    edge_labels = nx.get_edge_attributes(G, 'weight')
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=200,
        font_size=12,
        font_weight='bold',
        edge_color='#999999',
        width=2
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return node_color_map # return color mapping


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



path = r'C:\Users\tbeene\Desktop\Data\DUT\DUT_Simple.csv'
df_DUT = pd.read_csv(path)

# Drop columns where all values are NaN
df_DUT = df_DUT.dropna(axis=1, how='all')

# Replace remaining NaN values with 0
df_DUT = df_DUT.fillna(0)


df1, cov = Load_Data(50)

df = generate_multivariate_data(df1, 50)
# df = df_DUT
# cov = df_DUT.corr()


#plt.style.use('science')

corr_matrix = df.corr().replace(np.nan, 0) #.T.iloc[:,:-1].T
#corr_matrix = np.cov(df.T.replace(np.nan, 0))
d = len(df.T)
n = len(df)


Corr_0 = np.identity(d) + 10e-6 * np.random.rand(d,d)
distance_matrix = corr_matrix.values 

mst, all_edges = max_spanning_tree_edges(np.abs(distance_matrix))

# # pruned_mst, entropy_list, U_list, F_list = recursively_prune_edges(mst, len(distance_matrix)) 














from scipy.stats import sem, t

# Set number of runs
n_runs = 10

# Storage for each algorithm
scores_sim_all = []
scores_tabu_all = []
scores_swap_all = []
times_sim_all = []
times_tabu_all = []
times_swap_all = []

for ii in range(n_runs):
    pruned_mst, scoresim, time_listsim = simulated_annealing_prune(mst, len(distance_matrix), all_edges, distance_matrix, df)
    _, scoretabu, time_listtabu = tabu_search_prune(mst, len(distance_matrix), all_edges, distance_matrix, df)
    _, scoregreed, time_listgreed = greedy_edge_removal(mst, len(distance_matrix), all_edges, distance_matrix, df)
    
    scoresim = -(-np.array(scoresim))
    scoretabu = -(-np.array(scoretabu))
    scoregreed = -(-np.array(scoregreed))

    scores_sim_all.append(np.interp(np.linspace(0, time_listsim[-1], 100), time_listsim, scoresim))
    scores_tabu_all.append(np.interp(np.linspace(0, time_listtabu[-1], 100), time_listtabu, scoretabu))
    
# Convert to numpy arrays
scores_sim_all = (np.array(scores_sim_all))
scores_tabu_all = (np.array(scores_tabu_all))

# Mean and 95% confidence interval
mean_sim = np.mean(scores_sim_all, axis=0)
conf_sim = t.ppf(0.975, df=n_runs-1) * sem(scores_sim_all, axis=0)

mean_tabu = np.mean(scores_tabu_all, axis=0)
conf_tabu = t.ppf(0.975, df=n_runs-1) * sem(scores_tabu_all, axis=0)

# Common time axis
time_axis = np.linspace(0, max(time_listsim[-1], time_listsim[-1]), 100)

# Plot
plt.figure(figsize=(10, 6))
plt.rc('font', size=14)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=16)
plt.plot(time_axis, mean_sim, label='Simulated Annealing')
plt.fill_between(time_axis, mean_sim - conf_sim, mean_sim + conf_sim, alpha=0.3)

plt.plot(time_axis, mean_tabu, label='Tabu Search')
plt.fill_between(time_axis, mean_tabu - conf_tabu, mean_tabu + conf_tabu, alpha=0.3)

# plt.plot(time_axis, mean_swap, label='Swap Search')
# plt.fill_between(time_axis, mean_swap - conf_swap, mean_swap + conf_swap, alpha=0.3)

plt.plot(time_listgreed, scoregreed, label='Greedy Search', color='black')


#plt.ylim(0, 300)
plt.xlim(0,5)
plt.legend()
plt.ylabel(r'$\mathcal{L}$ [-]')
plt.xlabel('Time [s]')
plt.grid()
plt.show()






























# for ii in range(2):
#     pruned_mst1, scoregreed, time_listgreed =  greedy_edge_removal(mst, len(distance_matrix), all_edges, distance_matrix, df)
#     pruned_mst2, scoresim, time_listsim =  simulated_annealing_prune(mst, len(distance_matrix), all_edges, distance_matrix, df)
#     pruned_mst, scoretabu, time_listtabu =  tabu_search_prune(mst, len(distance_matrix), all_edges, distance_matrix, df)
    
    
    
#     print(label_nodes_by_cluster_list(pruned_mst, len(distance_matrix)))
    
#     labelsDBSCAN = label_nodes_by_cluster_list(pruned_mst, len(distance_matrix))
    
    
#     # plt.plot(entropy_list, label = 'S')
#     # plt.plot(U_list, label = 'U')
    
#     plt.plot(time_listsim, scoresim, label = 'Simulated annealing')
#     plt.plot(time_listtabu, scoretabu, label = 'Tabu search')

# plt.plot(time_listgreed, scoregreed,  label = 'Greedy search')
# plt.legend()
# plt.ylabel('$\sum |C_g|$ [-]')
# plt.xlabel('Time [s]')
# plt.grid()

# plt.style.use('science')
node_color_map = visualize_mst(len(df.T), pruned_mst)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(distance_matrix, cmap='coolwarm', center=0, cbar=True)

# Hide tick labels
ax.set_xticks([])
ax.set_yticks([])

# Get number of nodes
num_nodes = len(distance_matrix)

# Size of each heatmap cell
cell_size = 1.0

# Colored top bar (x-axis)
for i in range(num_nodes):
    color = node_color_map.get(i, 'white')
    rect = plt.Rectangle(
        (i, -0.6), # x, y
        cell_size, 1, # width, height (thin, long)
        facecolor=color,
        linewidth=0,
        transform=ax.transData,
        clip_on=False
    )
    ax.add_patch(rect)

# Colored left bar (y-axis)
for i in range(num_nodes):
    color = node_color_map.get(i, 'white')
    rect = plt.Rectangle(
        (-0.6, i), # x, y
        1, cell_size, # width, height
        facecolor=color,
        linewidth=0,
        transform=ax.transData,
        clip_on=False
    )
    ax.add_patch(rect)

# Format colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)

plt.tight_layout()
plt.show()







