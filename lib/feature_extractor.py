import torch
import torch.nn as nn
import torch.nn.functional as F

'''
premiere: custom activation function
'''
# class CustomActivationFunction(nn.Module): 
#     def __init__(self, beta=0.8, mode='swish'):
#         super().__init__()
#         self.beta = beta
#         self.mode = mode
#     def forward(self, x): 
#         if self.mode == 'swish':
#             return x * torch.sigmoid(self.beta * x)
#         elif self.mode == 'selu':
#             return nn.SELU()(x)
#         elif self.mode == 'relu':
#             return nn.ReLU()(x)

class Swish(nn.Module):
    def __init__(self, beta=0.8):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
'''
Feature extractors:
    Extract features of each node according to spatial neighbors and temporal histories
    Spatial neighbors: Graph Aggregation Layer or GCN
    Temporal histories: linear combination of weights. LSTM? GRU?
'''
# GCN convolution function x = A @ x
def spatial_gcn_aggregation(x:torch.Tensor, nearest_nodes:torch.Tensor, nearest_dist:torch.Tensor, n_heads, device, sigma):
    '''
    GCN weights: exp(-dij^2 / sigma^2)
    x: (B, T, N, C)
    nearest_nodes: (N, k+1) (self included)
    nearest_dist: (N, k+1)
    '''
    B, T, n_nodes, n_in = x.size(0), x.size(1), x.size(2), x.size(-1)
    # pad x
    pad_x = torch.zeros_like(x[:,:,0]).unsqueeze(2)
    pad_x = torch.cat((x, pad_x), dim=2)
    # different weights head
    lambda_ = torch.arange(1, n_heads+1, 1, dtype=torch.float32).to(device) / n_heads
    # comput graph weights
    nearest_dist, nearest_nodes = nearest_dist.view(-1), nearest_nodes.view(-1) # in (N*k,)
    weights = torch.exp(-(nearest_dist[:,None]**2) * lambda_ / (sigma**2)) # in (N*k, n_heads)
    weights[nearest_nodes== -1,:] = 0
    weights[weights < 1e-5] = 0
    assert not weights.isnan().any(), "GCN weights contain NaN"

    # compute graph aggregation
    if x.ndim == 4: # (B, T, N, C)
        agg = (pad_x[:,:,nearest_nodes, None] * weights[:,:,None]).view(B, T, n_nodes, -1, n_heads, n_in).sum(3) # added self with distance 0
    elif x.ndim == 5: # multihead x, in (B, T, N, n_heads, C)
        agg = (pad_x[:,:,nearest_nodes] * weights[:,:,None]).view(B, T, n_nodes, -1, n_heads, n_in).sum(3)
    else:
        raise ValueError(f"Invalid tensor shape: {x.shape}, dimension needs to be 4 or 5")
    
    assert not agg.isnan().any(), "GCN aggregation contain NaN"

    # distance convolution
    nearest_dist[nearest_dist == torch.inf] = 0
    dist_agg = (weights * nearest_dist[:,None]).view(n_nodes, -1, n_heads).sum(1) # in (N, n_heads)
    return agg, dist_agg

'''
Only Spatial Features
'''

class SpatialGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, nearest_nodes, nearest_dist, n_heads, device, sigma):
        super(SpatialGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nearest_nodes = nearest_nodes
        self.nearest_dist = nearest_dist
        self.n_heads = n_heads
        self.device = device
        self.sigma = sigma

        # Define the weights for the linear transformation
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        '''
        x: (batch_size, T, num_nodes, in_features)
        return: (batch_size, T, num_nodes, n_heads, out_features)
        '''

        # Apply the linear transformation
        x = self.linear(x)  # (batch_size, num_nodes, out_features)

        # Graph convolution using adjacency matrix
        agg, dist_agg = spatial_gcn_aggregation(x, self.nearest_nodes, self.nearest_dist, self.n_heads, self.device, self.sigma)  # (batch_size, T, num_nodes, n_heads, out_features), (N, n_heads)

        # add aggregation, other option concat
        return agg + dist_agg[None, None, :,:,None]  # Combine graph convolution and distance aggregation


class GraphSAGELayer(nn.Module):
    '''
    GAL: use the k nearest neighbors to aggregate the features of each node
    Use linear layer for aggregation, with custom activation function
    '''
    def __init__(self, n_in, n_out, nearest_nodes, n_heads, in_heads, device, use_out_fc=False, use_multihead_fc=True, use_single_fc=True):
        super(GraphSAGELayer, self).__init__()
        self.nearest_nodes = nearest_nodes # in (N, k+1)
        self.n_nodes = nearest_nodes.size(0) # N
        self.k = nearest_nodes.size(1) - 1 # k nearest neighbors
        self.n_heads = n_heads
        self.in_heads = in_heads
        self.device = device
        self.use_out_fc = use_out_fc
        self.use_multihead_fc = use_multihead_fc
        self.use_single_fc = use_single_fc # use single linear layer for all heads

        if self.use_single_fc:
            if self.use_multihead_fc: # use all in heads for n heads cauculation
                self.agg_fc = nn.Linear(self.in_heads * (self.k + 1), self.n_heads)
            else: # number of heads unchanged, cauculate each head features seperately and uniformly
                self.agg_fc = nn.Linear(self.k + 1, 1)
        else:
            self.agg_fc = nn.Linear(self.k + 1, 1)
            if self.use_multihead_fc:
                self.swish1 = Swish()
                self.multihead_fc = nn.Linear(self.in_heads, self.n_heads)
        
        if self.use_out_fc:
            self.swish2 = Swish()
            self.out_fc = nn.Linear(n_in, n_out)
        
        # self.alpha = alpha
        # self.use_relu = use_relu
        # self.relu = CustomActivationFunction()

    def forward(self, x):
        '''
        Spatial feature extraction
        x: (B, T, N, C) or (B, T, N, n_heads, C), c = n_in
        output: (B, T, N, n_heads, n_out)
        '''
        assert not x.isnan().any(), "Input x contains NaN"
        B, T, N, C = x.size(0), x.size(1), x.size(2), x.size(-1)
        # pad x
        pad_x = torch.zeros_like(x[:,:,0]).unsqueeze(2)
        pad_x = torch.cat((x, pad_x), dim=2)  # (B, T, N + 1, C)

        # align to fc dimensions
        if pad_x.ndim == 4:
            pad_x = pad_x.unsqueeze(-2) # in (B, T, N + 1, n_heads=1, C)
        in_heads = pad_x.size(-2)
        assert not torch.isnan(self.agg_fc.weight).any(), "Aggregation weights are None or contain NaN values"
        assert not torch.isinf(self.agg_fc.weight).any(), "Aggregation weights are None or contain inf values"

        if self.use_single_fc:
            if self.use_multihead_fc: # (k+1) * in_heads -> n_heads
                x_nn = pad_x[:,:,self.nearest_nodes.view(-1)].reshape(B, T, N, -1, C) # in (B, T, N, (k+1) * in_heads, C)
                x_agg = self.agg_fc(x_nn.transpose(-1, -2)).transpose(-1, -2) # in (B, T, N, n_heads, C)
            else: # k+1 -> 1
                x_nn = pad_x[:,:,self.nearest_nodes.view(-1)].reshape(B, T, N, -1, in_heads, C) # in (B, T, N, (k+1), in_heads, C)
                x_agg = self.agg_fc(x_nn.transpose(-1, -3)).squeeze(-1).transpose(-1, -2) # in (B, T, N, in_heads, C)
            assert not x_agg.isnan().any(), "Aggregation contain NaN"

        else:
            x_nn = pad_x[:,:,self.nearest_nodes.view(-1)].reshape(B, T, N, -1, in_heads, C)
            x_agg = self.agg_fc(x_nn.transpose(-1, -3)).squeeze(-1) # in (B, T, N, C, in_heads)
            if self.use_multihead_fc:
                x_agg = self.swish1(x_agg)
                x_agg = self.multihead_fc(x_agg).transpose(-1, -2) # in (B, T, N, n_heads, C)
        
        if self.use_out_fc:
            x_agg = self.swish2(x_agg)
            x_agg = self.out_fc(x_agg) # in (B, T, N, n_heads, n_out)
        
        return x_agg # in (B, T, N, n_heads, n_out)
    
class GNNExtrapolation(nn.Module):
    def __init__(self, n_nodes, t_in, T, nearest_nodes, nearest_dists, n_heads, device, sigma_ratio=400, alpha=0.2):
        super(GNNExtrapolation, self).__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.t_in = t_in
        self.T = T
        self.nearest_nodes = nearest_nodes
        self.nearest_dists = nearest_dists
        self.sigma = self.nearest_dists.max() / sigma_ratio
        # shrink function
        self.shrink = nn.Linear(t_in * n_heads, T - t_in)
        self.swish = Swish()

        # aggregate time features
        self.alpha = alpha
    
    def forward(self, x):
        '''
        x in (B, t_in, N, C)
        Step 1: convolution with the weighted graph, get multihead aggregations (B, T, N, n_heads, C)
        Step 2: temporal feature aggregation
        Step 3: shrink to guess size
        '''
        B, t_in, N, C = x.size()
        agg, _ = spatial_gcn_aggregation(x, self.nearest_nodes, self.nearest_dists, self.n_heads, self.device, self.sigma) # in (B, t_in, N, n_heads, C)
        agg = agg.permute(0, 2, 4, 1, 3).reshape(B, N, C, -1) # in (B, N, C, t_in * n_heads)
        y = self.shrink(agg).permute(0, 3, 1, 2) # in (B, T-t_in, N, C)
        y = self.swish(y)
        return torch.cat([x, y], dim=1)

    
class TemporalHistoryLayer(nn.Module):
    def __init__(self, in_features, out_features, interval):
        super(TemporalHistoryLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.interval = interval

        # Define the weights for the linear transformation
        self.linear = nn.Linear(interval * in_features, out_features)

    def forward(self, x):
        # x: (batch_size, T, num_nodes, *n_heads, in_features)
        B, T, N = x.size(0), x.size(1), x.size(2)
        pad_x = torch.zeros_like(x[:,0:self.interval])# .repeat(1, self.interval, 1, 1)
        pad_x = torch.cat((pad_x, x), dim=1)  # (batch_size, T + interval, num_nodes, in_features)
        input = torch.stack([pad_x[:,i:i+self.interval] for i in range(T)], dim=1)  # (batch_size, T, interval, num_nodes, in_features)
        if x.ndim == 4:
            input = input.transpose(2, 3).reshape(B, T, N, -1)  # (batch_size, T, num_nodes, in_features, interval)
        elif x.ndim == 5:
            n_heads = x.size(-2)
            input = input.transpose(2,3).reshape(B, T, N, n_heads, -1)

        # Apply the linear transformation
        output = self.linear(input)  # (batch_size, T, num_nodes, *n_heads, out_features)

        return output
    
class GraphSAGEExtrapolation(nn.Module):
    def __init__(self, n_nodes, t_in, T, nearest_nodes, n_in, n_heads, device, interval, n_layers=2):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.t_in = t_in
        self.T = T
        self.n_in = n_in
        self.nearest_nodes = nearest_nodes
        self.interval = interval

        self.input_layer = nn.Sequential(
            GraphSAGELayer(self.n_in, self.n_in, self.nearest_nodes, self.n_heads, 1, self.device, use_out_fc=False),
            Swish(),
            TemporalHistoryLayer(self.n_in, self.n_in, self.interval),
            Swish()
            )
        self.input_swish = Swish()
        self.n_layers = n_layers
        if self.n_layers > 1:
            self.SAGEs = nn.Sequential(
                *[
                    nn.Sequential(
                        GraphSAGELayer(self.n_in, self.n_in, self.nearest_nodes, self.n_heads, self.n_heads, self.device, use_out_fc=False, use_multihead_fc=False),
                        Swish(),
                        TemporalHistoryLayer(self.n_in, self.n_in, self.interval),
                        Swish()
                    ) for i in range(self.n_layers - 1)
                ]
            )
        
        self.shrink = nn.Linear(t_in * n_heads, T - t_in)
        self.swish = Swish()
    
    def forward(self, x):
        '''
        x in (B, t_in, N, C)
        output of input layer: (B, t_in, N, n_heads, C)
        # temporal feature aggregation: alpha
        output of GNN: (B, t_in, N, n_heads, C)
        '''
        B, t_in, N, C = x.size()
        agg = self.input_layer(x)
        if self.n_layers > 1:
            agg = self.SAGEs(agg) # in (B, t_in, N, n_heads, C)
        agg = agg.permute(0,2,4,1,3).reshape(B, N, C, -1)
        y = self.shrink(agg).permute(0,3,1,2)
        y = self.swish(y)
        return torch.cat([x, y], dim=1)
    
class FElayer(nn.Module):
    def __init__(self, in_features, out_features, nearest_nodes, n_heads, device, interval, parallel=True):
        super(FElayer, self).__init__()
        self.nearest_nodes = nearest_nodes
        self.n_heads = n_heads
        self.device = device
        # self.temporal_hist = TemporalHistoryLayer(in_features, out_features, interval)
        self.swish1 = Swish()
        self.swish2 = Swish()
        self.parallel = parallel
        if self.parallel:
            self.graph_sage = GraphSAGELayer(in_features, out_features, self.nearest_nodes, self.n_heads, 1, self.device, use_out_fc=True)
            self.temporal_hist = TemporalHistoryLayer(in_features, out_features * n_heads, interval)
        else:
            self.graph_sage = GraphSAGELayer(in_features, out_features, self.nearest_nodes, self.n_heads, 1, self.device, use_out_fc=True)
            self.temporal_hist = TemporalHistoryLayer(out_features, out_features, interval)
    def forward(self, x):
        '''
        input: embedded signals or signals itself
        output: concatenate of spatial and temporal features
        '''
        B, T, N, C = x.size()
        spatial_features = self.graph_sage(x) # in (B, T, N, h, C_out)
        spatial_features = self.swish1(spatial_features)
        if self.parallel:
            # concate features
            temporal_features = self.temporal_hist(x).unsqueeze(-2).reshape(B, T, N, self.n_heads, -1) # in (B, T, N, h, C_out)
            temporal_features = self.swish2(temporal_features)
            features = spatial_features + temporal_features
            # features = torch.cat([spatial_features, temporal_features], dim=-1) # in (B, T, N, h, C_out * 2)

        else:
            # GraphSAGE -> TH
            features = self.temporal_hist(spatial_features) # in (B, T, N, h, C_out)
            features = self.swish2(features)
        return features
    
class FeatureExtractor(nn.Module):
    '''
    extract temporal and spatial features, then concat / add
    '''
    def __init__(self, in_features, out_features, nearest_nodes, n_heads, device, interval):
        super(FeatureExtractor, self).__init__()
        self.nearest_nodes = nearest_nodes
        self.n_heads = n_heads
        self.device = device
        self.graph_sage = GraphSAGELayer(in_features, out_features, self.nearest_nodes, self.n_heads, 1, self.device, use_out_fc=True)
        # self.temporal_hist = TemporalHistoryLayer(in_features, out_features, interval)
        self.temporal_hist = TemporalHistoryLayer(out_features, out_features, interval)
        self.swish1 = Swish()
        self.swish2 = Swish()

    def forward(self, x):
        # x: (batch_size, num_nodes, in_features)
        # adj: (batch_size, num_nodes, num_nodes)
        B, T, N, C = x.size()
        # Extract spatial features
        spatial_features = self.graph_sage(x)  # (batch_size, num_nodes, n_heads, out_features)
        spatial_features = self.swish1(spatial_features)
        features = self.temporal_hist(spatial_features)
        features = self.swish2(features)
        return features # temporal_features
        # Extract temporal features
        # temporal_features = self.temporal_hist(x).unsqueeze(-2)# .reshape(B, T, N, self.n_heads, -1)  # (batch_size, num_nodes, n_heads, out_features)

        # return spatial_features + temporal_features  # Combine spatial and temporal features