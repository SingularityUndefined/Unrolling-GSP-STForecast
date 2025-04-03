import torch
def directed_graph_from_features(GL, features):
    '''
    Args:
        features (torch.Tensor) in (-1, T, n_nodes, n_heads, n_channels)
    Return:
        u_edges in (-1, T, GL.interval, n_nodes, n_heads, n_channels), with a lower triangular mask
    '''
    B, T, C = features.size(0), features.size(1), features.size(-1)
    # father features
    # father_features = features# .unsqueeze(2) # in (B, T, 1, n_nodes, n_heads, n_channels)
    # children features
    indice = torch.arange(0,GL.T).reshape(-1,1) - torch.arange(1, GL.interval + 1) # in (T, interval)
    features_i = features[:,indice.view(-1)].view(B, T, GL.interval, GL.n_nodes, -1, GL.n_channels) # in (B, T, interval, N, n_heads, n_channels)

    # multiply with Qs
    if GL.shared_params:
        Q_i = torch.einsum('hij, btvnhj -> btvnhi', GL.multiQ1, features_i)
        Q_j = torch.einsum('hij, btnhj -> btnhi', GL.multiQ2, features)
    else:
        Q_i = torch.einsum('thij, btvnhj -> btvnhi', GL.multiQ1, features_i)
        Q_j = torch.einsum('thij, btnhj -> btnhi', GL.multiQ2, features)
    # assertation
    assert not torch.isnan(Q_j).any(), f'Q_j has NaN value: Q2 in ({GL.multiQ2.max().item():.4f}, {GL.multiQ2.min().item():.4f}; features in ({features.max().item()}, {features.min().item()}))'
    assert not torch.isnan(Q_i).any(), f'Q_i has NaN value: Q1 in ({GL.multiQ1.max().item():.4f}, {GL.multiQ1.min().item():.4f}, features in ({features.max()}, {features.min()})'
    # multiply two qs
    weights = torch.exp(Q_i * Q_j.unsqueeze(2)).sum(-1) # in (B, T, interval, N, n_heads)

    # mask
    mask = torch.ones(T, GL.interval).tril_(diagonal=-1).unsqueeze(0).unsqueeze(3).unsqueeze(4).to(GL.device) # .repeat(B, 1, 1, GL.n_nodes, GL.n_heads).to(GL.device) # in (B, T, interval, N, n_heads)
    weights = weights * mask

    # normalization
    in_degree = weights.sum(2) # in (B, T, N, n_heads)
    inv_in_degree = torch.where(in_degree > 0, torch.ones((1,), device=GL.device) / in_degree, torch.zeros((1,), device=GL.device))
    inv_in_degree = torch.where(inv_in_degree == torch.inf, torch.zeros((1), device=GL.device), inv_in_degree)
    inv_in_degree = torch.where(inv_in_degree == torch.nan, torch.zeros((1), device=GL.device), inv_in_degree)
    weights = weights * inv_in_degree.unsqueeze(2) # in (B, T, interval, N, n_heads)
    # print('weights', weights.max(), weights.min(), torch.isnan(weights).any())
    # print('weights', weights.shape, weights.max(), weights.min(), torch.isnan(weights).any())
    return weights

GL = type('GL', (object,), {})()
GL.T = 24
GL.interval = 3
GL.n_nodes = 15
GL.n_heads = 4
GL.n_channels = 3
GL.shared_params = True
GL.device = 'cpu'
GL.multiQ1 = torch.randn(GL.n_heads, GL.n_channels, GL.n_channels)
GL.multiQ2 = torch.randn(GL.n_heads, GL.n_channels, GL.n_channels)
features = torch.randn(2, GL.T, GL.n_nodes, GL.n_heads, GL.n_channels)
# Call the function
u_edges = directed_graph_from_features(GL, features)
print(u_edges.shape)  # Expected shape: (B, T, interval, n_nodes, n_heads)
print(u_edges[0,:,:,0,0])
# Test the function with a sample input
# features = torch.randn(2, GL.T, GL.n_nodes, GL.n_heads, GL.n_channels)
# u_edges = directed_graph_from_features(GL, features)
# print(u_edges.shape)  # Expected shape: (B, T, interval, n_nodes, n_heads)
