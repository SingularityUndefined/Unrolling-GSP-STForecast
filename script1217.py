import networkx as nx
import heapq

def k_closest_nodes(G, node, k):
    # 使用Dijkstra算法找到所有节点到目标节点的最短路径长度
    lengths = nx.single_source_dijkstra_path_length(G, node)
    
    # 使用堆排序找到距离最短的k个节点
    closest_nodes = heapq.nsmallest(k + 1, lengths.items(), key=lambda x: x[1])
    
    # 移除自身节点
    closest_nodes = [n for n, _ in closest_nodes if n != node]
    
    return closest_nodes# [:k]

# 创建有向图
G = nx.DiGraph()
G.add_edges_from([
    (1, 2), (1, 3), (2, 3), (2, 4),
    (3, 4), (4, 5), (5, 1), (5, 6)
])

# 找到节点1的距离最近的3个节点（不包含节点1）
closest_nodes = k_closest_nodes(G, 1, 3)
print(f"Closest 3 nodes to node 1: {closest_nodes}")
