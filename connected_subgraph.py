from collections import defaultdict

def find_connected_subgraphs(n_nodes, edges):
    """
    Find the connected subgraphs in a directed graph and return the nodes in each subgraph.

    Args:
    - n_nodes (int): Number of nodes in the graph.
    - edges (list of list): List of directed edges, where edges[i] = [u, v].

    Returns:
    - list of list: Each sublist contains the indices of the nodes in a connected subgraph.
    """

    # Build adjacency list for the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # Since we need connected components, treat it as undirected.

    visited = set()
    connected_subgraphs = []

    def dfs(node, component):
        """Perform DFS to collect all nodes in the current connected component."""
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                stack.extend(graph[current])

    # Iterate over all nodes to find connected components
    for node in range(n_nodes):
        if node not in visited:
            component = []
            dfs(node, component)
            connected_subgraphs.append(component)

    return connected_subgraphs

# Example Usage:
# Number of nodes and edges
n_nodes = 6
edges = [[0, 1], [1, 2], [3, 4]]  # Example graph

# Find the nodes in each connected subgraph
result = find_connected_subgraphs(n_nodes, edges)
print("Connected subgraphs (node indices):", result)

import numpy as np

def calculate_laplacian_matrices(weights, subgraph_indices):
    """
    计算子图的拉普拉斯矩阵。
    
    Args:
    - weights (numpy.ndarray): 图的权重矩阵，形状为 (n_nodes, k, n_heads)。
    - subgraph_indices (list of list): 每个子图的节点索引。

    Returns:
    - list of numpy.ndarray: 每个子图的拉普拉斯矩阵。
    """
    n_nodes, k, n_heads = weights.shape
    laplacians = []

    for indices in subgraph_indices:
        # 提取子图的节点数
        num_nodes = len(indices)
        
        # 初始化邻接矩阵
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        # 提取子图中的权重
        for i, node in enumerate(indices):
            for neighbor_idx in range(k):
                neighbor = weights[node, neighbor_idx, :]
                if neighbor.any():  # 检查是否为零填充
                    for head in range(n_heads):
                        neighbor_node = int(neighbor[head])
                        if neighbor_node in indices:
                            j = indices.index(neighbor_node)
                            adjacency_matrix[i, j] += 1  # 假设头的值合适相当
                
#APP  LOG                 


# 下面是完整的代码实现，包括权重提取、邻接矩阵构建和拉普拉斯矩阵计算。

### **Python 实现（完整）**

import numpy as np

def calculate_laplacian_matrices(weights, subgraph_indices):
    """
    计算子图的拉普拉斯矩阵。
    
    Args:
    - weights (numpy.ndarray): 图的权重矩阵，形状为 (n_nodes, k, n_heads)，
                               每个元素表示一个邻接节点的索引和对应权重。
    - subgraph_indices (list of list): 每个子图的节点索引。

    Returns:
    - list of numpy.ndarray: 每个子图的拉普拉斯矩阵。
    """
    n_nodes, k, n_heads = weights.shape
    laplacians = []

    for indices in subgraph_indices:
        # 提取子图的节点数
        num_nodes = len(indices)
        
        # 初始化邻接矩阵
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        # 提取子图中的权重
        for i, node in enumerate(indices):
            for neighbor_idx in range(k):
                for head in range(n_heads):
                    # 读取邻接节点索引和权重
                    neighbor_node = weights[node, neighbor_idx, head]
                    if neighbor_node != 0 and neighbor_node in indices:
                        j = indices.index(neighbor_node)
                        adjacency_matrix[i, j] += 1  # 假设权重为1，实际权重可以修改

        # 计算度矩阵
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        
        # 计算拉普拉斯矩阵
        laplacian_matrix = degree_matrix - adjacency_matrix
        laplacians.append(laplacian_matrix)

    return laplacians

# 示例用法
if __name__ == "__main__":
    # 输入权重矩阵 (n_nodes, k, n_heads)
    weights = np.array([
        [[1, 2], [3, 0], [0, 0]],  # 节点 0 的邻接节点
        [[0, 3], [2, 0], [0, 0]],  # 节点 1 的邻接节点
        [[1, 3], [0, 0], [0, 0]],  # 节点 2 的邻接节点
        [[1, 2], [0, 0], [0, 0]],  # 节点 3 的邻接节点
    ])
    # 子图节点索引
    subgraph_indices = [
        [0, 1, 2],  # 第一个子图包含节点 0, 1, 2
        [3]         # 第二个子图包含节点 3
    ]

    # 计算拉普拉斯矩阵
    laplacians = calculate_laplacian_matrices(weights, subgraph_indices)

    # 输出结果
    for i, laplacian in enumerate(laplacians):
        print(f"子图 {i+1} 的拉普拉斯矩阵:")
        print(laplacian)