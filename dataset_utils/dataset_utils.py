# This file is for data file structures
import pandas as pd
import numpy as np
import os 
import pickle
from collections import Counter


# generate adj matrix from df
def adj_matrix_from_distance(df:pd.DataFrame, sensor_dict:dict | None=None, squared_dist=False, self_loop=False):
    # sensor_dict[station_name] = station_index
    # df in ['from', 'to', 'cost']

    # get all nodes
    n_nodes = max(max(df['from'].values), max(df['to'].values)) + 1
    if sensor_dict is None:
        from_list, to_list = list(df['from'].values), list(df['to'].values)
    else:
        from_list = [sensor_dict[i] for i in df['from'].values]
        to_list = [sensor_dict[i] for i in df['to'].values]
    
    u_edges = np.array([from_list + to_list, to_list + from_list]).T
    n_edges = len(from_list) * 2
    dic = Counter([(u_edges[i,0], u_edges[i,1]) for i in range(n_edges)])
    assert max(list(dic.values())), 'distance graph asymmetric'
    # distance matrix
    dists = df[df.columns[-1]].values
    mean_dist = dists.mean()
    std_dist = dists.std()
    if not squared_dist:
        weights = np.exp(-dists / mean_dist)
    else:
        weights = np.exp(- (dists ** 2) / (std_dist ** 2))

    weights[weights < 1e-8] = 0
    if self_loop:
        adj = np.eye(n_nodes)
    else:
        adj = np.zeros((n_nodes, n_nodes))
    adj[from_list, to_list] = weights
    adj[to_list, from_list] = weights

    return adj
    
# generate distance df from adj matrix
def df_from_adj_matrix(adj:np.ndarray, squared_dist=None):
    # adj: n * n matrix, symetric
    # return: from, to, cost
    df = pd.DataFrame(columns=['from', 'to', 'cost'])
    # compute real distances
    n_nodes = adj.shape[0]
    for i in range(n_nodes):
        for j in range(i):
            if adj[i,j] > 0:
                if squared_dist:
                    dist = np.sqrt(-np.log(adj[i,j]))
                else:
                    dist = -np.log(adj[i,j])
                df = df.append({'from': i, 'to': j, 'cost': dist}, ignore_index=True)
    
    return df


# save df as csv or pickle
def save_cost_df(df:pd.DataFrame, file_path):
    form = file_path.split('.')[-1]
    assert form in ['csv', 'pkl']
    if form == 'csv':
        df.to_csv(file_path, index=False)
    else:
        df.to_pickle(file_path)

# save adj matrix as pickle
def save_adj_matrix(adj, file_path):
    form = file_path.split('.')[-1]
    assert form in ['npy', 'pkl']
    if form == 'npy':
        np.save(file_path, adj)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump(adj, f)
