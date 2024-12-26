'''
This file is the scripts for all the codes
'''

'''
Task 1 Check all the connections in graph
'''
from dataloader import physical_graph, TrafficDataset
import numpy as np
import torch

def get_degrees(n_nodes, u_edges:torch.Tensor):
    '''
    u_edges: ndarray, in (n_edges, 2), already bidirectional
    '''
    n_edges = u_edges.size(0)
    degrees = np.zeros((n_nodes,), dtype=int)
    for i in range(n_edges):
        degrees[u_edges[i,0]] += 1
        # degrees[u_edges[i,1]] += 1
    return degrees

'''
Task 2 Get the K-nearest neighbors for connection
'''
import torch
import heapq
import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt

# 构建有向图

# 可视化有向图
def visualise_graph(G, dataset_name, fig_name):
    pos = nx.spring_layout(G)  # 使用弹簧布局来定位节点
    nx.draw(G, pos, with_labels=False, node_size=7, node_color='lightblue', arrowsize=2)
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=2)

    plt.title(dataset_name)
    plt.savefig(fig_name, dpi=800)

def find_k_nearest_neighbors(edges:torch.Tensor, distances:torch.Tensor, k):
    edges = edges.detach().cpu().numpy()
    dist = distances.detach().cpu().numpy()
    graph = nx.DiGraph()
    for i in range(len(edges)): 
        graph.add_edge(edges[i, 0], edges[i, 1], weight=dist[i])
    
    visualise_graph(graph, 'PeMS04', 'PeMS04.png')
    nearest_neighbors = {}
    # print('nodes', graph.nodes)

    for node in graph.nodes: # 使用 Dijkstra 算法计算从当前节点出发的最短路径 
        distances = nx.single_source_dijkstra_path_length(graph, node) # 将结果按距离排序，并获取最近的 N 个节点 
        closest_nodes = heapq.nsmallest(k, distances.items(), key=lambda x: x[1]) # 存储结果 
        # nearest_neighbors[node] = closest_nodes 
        nearest_neighbors[node] = ([i for (i,_) in closest_nodes], [j for (_,j) in closest_nodes])
    return nearest_neighbors


from utils import *
from collections import Counter

batch_size = 12
num_workers = 4
dataset_dir = '/mnt/qij/datasets/PEMS0X_data/'
experiment_name = 'k_hop'
dataset_name = 'PEMS07'
T = 12
t_in = 6
stride = 3

train_set, val_set, test_set, train_loader, val_loader, test_loader = create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers)

u_edges, u_dist = train_set.u_edges, train_set.u_distance
n_nodes = train_set.n_nodes
print(u_edges.shape, u_dist.shape)

degrees = get_degrees(n_nodes, u_edges)
print(degrees.max(), degrees.min())
print(Counter(degrees))

# CDF curve
def plot_cdf(dist:torch.Tensor, bins, dataset_name):
    plt.hist(dist.numpy(), bins, cumulative=True, density=True, histtype='step', label='CDF')
    # 添加标题和标签
    plt.grid(True)
    plt.title('Cumulative Distribution Function for ' + dataset_name)
    plt.xlabel('Data')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.savefig('cdf_' + dataset_name + '.png', dpi=800)

print(u_dist.shape)
plot_cdf(u_dist, 60, dataset_name)
# 创建示例数据

