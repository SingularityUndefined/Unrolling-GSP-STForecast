import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import math

import pandas as pd
import numpy as np
import torch.nn.functional as F
import networkx as nx
from lib.backup_modules import k_hop_neighbors, LR_guess, find_k_nearest_neighbors
# from statsmodels.tsa.api import VAR
# from statsmodels.tsa.stattools import adfuller
class Swish(nn.Module): 
    def __init__(self, beta=0.8):
        super().__init__()
        self.beta = beta
    def forward(self, x): 
        return x * torch.sigmoid(self.beta * x)

# node embedding
class GNNExtrapolation(nn.Module):
    '''GNN extrapolation
    '''
    def __init__(self, n_nodes, t_in, T, nearest_nodes, nearest_dists, n_heads, device, sigma_ratio=500):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.t_in = t_in
        self.T = T
        self.nearest_nodes = nearest_nodes
        self.nearest_dists = nearest_dists
        assert T > t_in, 't_in > T'
        # model in markovian
        # self.MLP = nn.Sequential(nn.Linear(t_in * n_heads, hidden_size), nn.ReLU(), nn.Linear(hidden_size, T - t_in), nn.ReLU())
        # self.shrink = nn.Linear(t_in * n_heads, T - t_in) 
        self.shrink = nn.Sequential(nn.Linear(t_in * n_heads, T - t_in), nn.SELU())
        self.sigma = self.nearest_dists.max() / sigma_ratio
        
    def forward(self, x):
        # signals in (Batch, T, n_nodes, n_channels)?
        B, t_in, n_nodes, n_channels = x.size()
        # n_nodes = n_nodes - 1
        # aggregation
        agg, _ = gcn_aggregation(x, self.nearest_nodes, self.nearest_dists, self.n_heads, self.device, self.sigma) # in (B, t_in, N, n_heads, n_channels)
        assert not torch.isnan(agg).any(), 'extrapolation agg has nan value'
        agg = agg.permute(0,2,4,1,3).reshape(B, n_nodes, n_channels, -1) # in (B, N, n_channels, t_in * n_heads)
        y = self.shrink(agg).permute(0,3,1,2)
        assert not torch.isnan(y).any(), f'weights has NaN :{torch.isnan(self.shrink[0].weight).any()}'
        # print('[x, y]', x.shape, y.shape)
        return torch.cat([x, y], dim=1)

class GALExtrapolation(nn.Module):
    '''GNN extrapolation, selu activation
    '''
    def __init__(self, n_nodes, t_in, T, nearest_nodes, n_in,
                 # nearest_dists,
                   n_heads, device):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.t_in = t_in
        self.T = T
        self.n_in = n_in
        # self.agg = GraphAggregationLayer()
        self.nearest_nodes = nearest_nodes
        # self.nearest_dists = nearest_dists
        assert T > t_in, 't_in > T' # SET RELU FALSE
        self.agg_layer = GraphAggregationLayer(self.n_in, self.n_in, self.nearest_nodes, self.n_heads, 1, self.device, 0, use_relu=False)
        # model in markovian
        # self.MLP = nn.Sequential(nn.Linear(t_in * n_heads, hidden_size), nn.ReLU(), nn.Linear(hidden_size, T - t_in), nn.ReLU())
        # self.shrink = nn.Linear(t_in * n_heads, T - t_in) 
        self.shrink = nn.Sequential(nn.Linear(t_in * n_heads, T - t_in), nn.SELU())
        # self.bn = nn.BatchNorm2d()
        # nn.init.xavier_uniform_(self.shrink[0].weight)
        # self.sigma = sigma
        
    def forward(self, x):
        # signals in (Batch, T, n_nodes, n_channels)?
        B, t_in, n_nodes, n_channels = x.size()
        # n_nodes = n_nodes - 1
        # aggregation
        # agg, _ = gcn_aggregation(x, self.nearest_nodes, self.nearest_dists, self.n_heads, self.device, self.sigma) # in (B, t_in, N, n_heads, n_channels)
        # agg = multihead_aggregation(x, self.n_heads, self.device)
        try:
            agg = self.agg_layer(x)
        except AssertionError as ae:
            print(f'Error in GALExtrapolation:agg_layer - {ae}')
        # print(agg.shape)

        # assert not torch.isnan(agg).any(), 'extrapolation agg has nan value'
        agg = agg.permute(0,2,4,1,3).reshape(B, n_nodes, n_channels, -1) # in (B, N, n_channels, t_in * n_heads)
        y = self.shrink(agg).permute(0,3,1,2)
        assert not torch.isnan(y).any(), f'weights has NaN :{torch.isnan(self.shrink[0].weight).any()}'
        # print('[x, y]', x.shape, y.shape)
        return torch.cat([x, y], dim=1)
##################################################################################

def multihead_aggregation(x:torch.Tensor, n_heads, device):
    lambda_ = torch.arange(1, n_heads + 1, 1, dtype=torch.float, device=device) / n_heads
    # no aggregation, only linear transform
    lambda_ = torch.exp(-lambda_ ** 2)
    x_agg = x.unsqueeze(-2) * lambda_.unsqueeze(-1) # in (B, T, N, n_heads, n_in)
    return x_agg

def gcn_aggregation(x:torch.Tensor, nearest_nodes:torch.Tensor, nearest_dist:torch.Tensor, n_heads, device, sigma=6):
    '''
    nearest_nodes: (N, k + 1) (self)
    nearest_dist: (N, k + 1)
    '''
    # nearest_nodes = nearest_nodes[:, 1:]
    # nearest_dist = nearest_dist[:, 1:]
    assert not torch.isnan(x).any(), 'x has NaN value'
    B, T, n_nodes, n_in = x.size(0), x.size(1), x.size(2), x.size(-1) # already padded
    # pad x
    pad_x = torch.zeros_like(x[:,:,0]).unsqueeze(2)
    pad_x = torch.cat((x, pad_x), dim=2)
    lambda_ = torch.arange(1, n_heads + 1, 1, dtype=torch.float, device=device) / n_heads # 
    # reshape
    # print(nearest_nodes.shape)
    # k = nearest_dist.size(1)
    nearest_dist, nearest_nodes = nearest_dist.view(-1), nearest_nodes.view(-1) # in (N *k)
    weights = torch.exp(- (nearest_dist[:,None] ** 2) * lambda_ / (sigma ** 2)) # in (N*k, n_heads)
    weights[nearest_nodes == -1,:] = 0
    # normalize weights?
    weights[weights < 1e-8] = 0

    assert not torch.isnan(weights).any(), 'GCN weights NaN'
    # print('weights < 1 max', weights[weights < 1].max(), weights[weights > 0].min())
    # # normalize?
    # degree = weights.view(n_nodes, -1, n_heads).sum(1, keepdim=True).repeat(1, k, 1).view(-1, n_heads)
    # inv_degree = torch.where(degree > 0, torch.ones((1), device=device) / degree, torch.zeros((1), device=device))
    # inv_degree = torch.where(inv_degree == torch.inf, 0, inv_degree)
    # weights = weights * inv_degree
    
    if x.ndim == 4:
        agg = (pad_x[:,:,nearest_nodes,None] * weights[:,:,None]).view(B, T, n_nodes, -1, n_heads, n_in).sum(3)
        # agg = agg + x.unsqueeze(3)
    else:
        agg = (pad_x[:,:,nearest_nodes] * weights[:,:,None]).view(B, T, n_nodes, -1, n_heads, n_in).sum(3)
        # agg = agg + x
    assert not torch.isnan(agg).any(), 'agg has NaN'
    # agg = agg + x.unsqueeze()
    nearest_dist[nearest_dist == torch.inf] = 0
    dist_agg = (weights * nearest_dist[:,None]).view(n_nodes, -1, n_heads).sum(1)
    assert not torch.isnan(dist_agg).any(), 'dist_agg has NaN'
    # print(dist_agg.max(), dist_agg.min())
    # pad agg
    # pad_agg = torch.zeros_like(agg[:,:,0], device=device).unsqueeze(2)
    # print(pad_agg.shape, agg.shape)
    # agg = torch.cat((agg, pad_agg), dim=2)
    return agg, dist_agg # in (B, T, N, n_heads, n_in), (N, n_heads)

class GraphAggregationLayer(nn.Module):
    def __init__(self, n_in, n_out, nearest_nodes, n_heads, in_heads, device, use_out_fc=True, use_multihead_fc=True, alpha=0.2, use_relu=True):
        super().__init__()
        self.nearest_nodes = nearest_nodes
        self.k = nearest_nodes.size(1) - 1
        self.n_nodes = nearest_nodes.size(0)
        self.n_heads = n_heads
        self.device = device
        self.in_heads = in_heads
        # self.agg_mat = Parameter(torch.ones((n_heads, self.k + 1, in_heads), device=self.device) / torch.sqrt(n_heads * (self.k + 1) * in_heads), requires_grad=True) # TODO:  can be reduced 
        # self.agg_fc = nn.Linear(self.k + 1, 1)
        # if in_heads == 1:
            # self.multi_fc = nn.Linear(1, n_heads)
        # self.bias = Parameter(torch.zeros(n_heads, device=self.device), requires_grad=True)
        self.agg_fc = nn.Linear(self.k + 1, 1)
        
        self.use_multihead_fc = use_multihead_fc
        if self.use_multihead_fc:
            self.swish1 = Swish()
            self.multihead_fc = nn.Linear(self.in_heads, self.n_heads)
        # self.agg_fc = nn.Linear(self.in_heads * (self.k + 1), self.n_heads)
        self.use_out_fc = use_out_fc
        if self.use_out_fc:
            self.swish2 = Swish()
            self.out_fc = nn.Linear(n_in, n_out)
        # nn.init.xavier_uniform_(self.agg_fc.weight)
        # nn.init.xavier_uniform_(self.out_fc.weight)
        self.alpha = alpha
        self.use_relu = use_relu
        self.relu = Swish() # nn.SELU()
        # self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        '''
        x in (B, T, n_nodes, n_heads*, n_in)
        '''
        # pad x
        assert not torch.isnan(x).any(), 'x has NaN value'
        B, T, n_in = x.size(0), x.size(1), x.size(-1)
        pad_x = torch.zeros_like(x[:,:,0]).unsqueeze(2)
        pad_x = torch.cat((x, pad_x), dim=2)
        # get signal neighbor list for all signals
        if pad_x.ndim == 4:
            pad_x = pad_x.unsqueeze(-2)
            
        head_in = pad_x.size(-2)
        # x_nn = pad_x[:,:, self.nearest_nodes.view(-1)].reshape(B, T, self.n_nodes, -1, n_in) # in (B, T, N, k * head_in, n_in) # .reshape(B, T, self.n_nodes, -1, head_in, n_in) # in (B, T, N, k, head_in, n_in)
        # x_agg = self.agg_fc(x_nn.transpose(-1, -2)).transpose(-1, -2) #TODO: reduce more parameters
        # assert not torch.isnan(x_agg).any(), f'x_agg has NaN value, agg_fc has NaN value {torch.isnan(self.agg_fc.weight).any()}'
        assert not torch.isinf(self.agg_fc.weight).any(), f'agg_fc.weight has INF value'
        # assert not torch.isinf(self.agg_fc.weight).any(), f'agg_fc_1 has inf value'

        x_nn = pad_x[:,:, self.nearest_nodes.view(-1)].reshape(B, T, self.n_nodes, -1, head_in, n_in)
        # print(x_nn.shape)
        x_agg = self.agg_fc(x_nn.transpose(-1, -3)).squeeze(-1)# .transpose(-1, -2)
        assert not torch.isnan(x_agg).any(), f'x_agg_1 has NaN value, agg_fc has NaN value {torch.isnan(self.agg_fc.weight).any()}'
        
        if self.use_multihead_fc:
            assert not torch.isinf(self.multihead_fc.weight).any(), 'multihead_fc.weight has INF value'
            assert not torch.isinf(self.multihead_fc.bias).any(), 'multihead_fc.bias has INF value'
            x_agg = self.swish1(x_agg)
            x_agg = self.multihead_fc(x_agg)# .transpose(-1, -2)
            assert not torch.isnan(x_agg).any(), f'x_agg has NaN value, agg_fc_2 has NaN value {torch.isnan(self.agg_fc_2.weight).any()}'
        
        x_agg = x_agg.transpose(-1, -2)
        
        if self.use_out_fc:
            assert not torch.isinf(self.out_fc.weight).any(), f'out_fc.weight has INF value'
            assert not torch.isinf(self.out_fc.bias).any(), f'out_fc.bias has INF value'
            x_agg = self.swish2(x_agg)
            x_agg = self.out_fc(x_agg)
            assert not torch.isnan(x_agg).any(), f'x_agg has NaN value, out_fc has NaN value {torch.isnan(self.out_fc.weight).any()}'
        # # custom linear layer
        # x_agg = torch.einsum('btnkhi, gkh -> btngi', x_nn, self.agg_mat) # in (B, T, n_nodes, n_heads, n_in)
        # x_agg = x_agg + self.bias[None, None, None, :, None]
        # cross-time aggregation 
        x_agg[:,1:] = (1 - self.alpha) * x_agg[:,1:] + self.alpha * x_agg[:,:-1]
        # activation
        if self.use_relu:
            x_agg = self.relu(x_agg)

        return x_agg

class GraphConvolutionLayer(nn.Module):
    def __init__(self, n_in, n_out, n_nodes, n_heads, nearest_nodes, nearest_dist, device, sigma=6, alpha=0.2, use_dist_conv=False):
        super().__init__()
        self.use_dist_conv = use_dist_conv
        if self.use_dist_conv:
            self.fc = nn.Linear(n_in + 1, n_out)
        else:
            self.fc = nn.Linear(n_in, n_out)
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.device = device
        self.nearest_nodes = nearest_nodes
        self.nearest_dist = nearest_dist
        self.sigma = sigma
        self.relu = nn.ReLU() # TODO: ReLU or SELU? as a conbination?
        self.alpha = alpha
        # self.use_dist_conv = use_dist_conv

    def forward(self, x):
        # aggregate
        B, T = x.size(0), x.size(1)
        agg, dist_agg = gcn_aggregation(x, self.nearest_nodes, self.nearest_dist, self.n_heads, self.device, self.sigma) # in (B, T, N, n_heads, n_in), (N, n_heads)
        # add next layer, agg x_i(t+1)
        if self.use_dist_conv:
            dist_agg = dist_agg[None, None, :,:].repeat(B, T, 1, 1).unsqueeze(-1)
            agg = torch.cat((agg, dist_agg), -1)
            # print('agg', agg.max(), agg.min())
        # else: use Laplacian embedding (spatial embeddings)
        # time axis aggregation
        agg[:,1:] = (1 - self.alpha) * agg[:,1:] + self.alpha * agg[:,:-1]

        out = self.fc(agg)
        return self.relu(out)

class FeatureExtractor(nn.Module):
    # def __init__(self, n_in, n_out, n_nodes, n_heads, nearest_nodes, nearest_dists, device, n_layers=3, sigma=6, alpha=0.2, use_dist_conv=False):
    def __init__(self, n_in, n_out, n_heads, nearest_nodes, device, n_layers=3, alpha=0.2, use_graph_agg=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.nearest_nodes = nearest_nodes
        # self.nearest_dists = nearest_dists
        self.alpha = alpha
        self.n_layers = n_layers
        self.device = device
        self.use_graph_agg = use_graph_agg

        if self.use_graph_agg:
            self.input_layer = GraphAggregationLayer(n_in, n_out, self.nearest_nodes, n_heads, 1, self.device, alpha=alpha)
            if self.n_layers > 1:
                self.GNN = nn.Sequential(*[GraphAggregationLayer(n_out, n_out, self.nearest_nodes, n_heads, n_heads, self.device, use_out_fc=False, use_multihead_fc=False, alpha=alpha) for i in range(n_layers - 1)])
        else:
            self.nn = nn.Sequential(nn.Linear(1, n_heads), nn.SELU()) # n_nodes, n_in -> n_nodes, n_head, n_in

        # self.input_layer = GraphConvolutionLayer(n_in, n_out, n_nodes, n_heads, self.nearest_nodes, self.nearest_dists, device, sigma, alpha, self.use_dist_conv)
        # GNN layers
        # if self.n_layers > 1:
        #     self.GNN = nn.Sequential(*[GraphConvolutionLayer(n_out, n_out, n_nodes, n_heads, self.nearest_nodes, self.nearest_dists, device, sigma, alpha, False)
        #         for i in range(n_layers - 1)
        #     ])

    def forward(self, x):
        if self.use_graph_agg:
            # print(x.shape)\
            try:
                out = self.input_layer(x)
            except AssertionError as ae:
                raise AssertionError(f'input layer - {ae}')
            assert not torch.isnan(out).any(), 'GCN Feature Extractor 1st layer NaN' 
        # print('GCN 1:', out.size())
            if self.n_layers == 1:
                return out
            else:
                return self.GNN(out)
        else:
            return self.nn(x.unsqueeze(-1)).transpose(-1, -2) # in (B, T, N, n_nodes, n_in)
            # use multihead_agg


# return k-hop edges 

class GraphLearningModule(nn.Module):
    '''
    learning the directed and undirected weights from features
    '''
    def __init__(self, T, n_nodes, nearest_nodes, n_heads, device, n_channels=None, sigma=6, Q1_init=1.2, Q2_init=0.8, M_init=1.5) -> None:
        '''
        Args:
            u_edges (torch.Tensor) in (n_edges, 2) # nodes regularized
            u_dist (torch.Tensor) in (n_edges)
        We construct d_edges by hand with n_nodes
        '''
        super().__init__()
        self.T = T
        self.n_nodes = n_nodes
        self.device = device
        # construct d_edges, d_dist
        self.nearest_nodes = nearest_nodes
        # multi_heads
        self.n_heads = n_heads

        # self.n_features = n_features # feature channels
        self.n_channels = n_channels
        self.n_out = self.n_out = (self.n_channels + 1) // 2
        # define multiM, multiQs
        self.Q1_init = Q1_init
        self.Q2_init = Q2_init
        self.M_init = M_init
        q_form = torch.zeros((self.n_heads, self.n_out, self.n_channels), device=self.device)
        q_form[:,:, :self.n_out] = torch.diag_embed(torch.ones((self.n_heads, self.n_out), device=self.device))
        # all variables shared across time
        self.multiQ1 = Parameter(q_form * self.Q1_init, requires_grad=True)
        self.multiQ2 = Parameter(q_form * self.Q2_init, requires_grad=True)
        self.multiM = Parameter(torch.diag_embed(torch.ones((self.n_heads, self.n_channels), device=self.device)) * self.M_init, requires_grad=True) # in (n_heads, n_channels, n_channels)

    def undirected_graph_from_features(self, features):
        '''
        Args:
            features (torch.Tensor) in (-1, T, n_nodes, n_heads, n_channels)
        Returns:
            u_edges in (-1, T, n_edges, n_heads)
        '''
        B, T = features.size(0), features.size(1)
        weights = {}

        # pad features
        # nn = self.nearest_nodes[:, 1:]
        pad_features = torch.zeros_like(features[:,:,0], device=self.device).unsqueeze(2)
        pad_features = torch.cat((features, pad_features), dim=2)

        feature_j = pad_features[:,:,self.nearest_nodes[:,1:].reshape(-1)].view(B, T, self.n_nodes, -1, self.n_heads, self.n_channels)
        # print(features.size(), feature_j.size())

        df = features.unsqueeze(3) - feature_j # in (B, T, N, k, n_heads, n_channels)
        Mdf = torch.einsum('hij, btnehj -> btnehi', self.multiM, df) # in (B, T, N, k, n_heads, n_channels)
        weights = torch.exp(- (Mdf ** 2).sum(-1)) # in (B, T, N, k, n_heads)
        # mask weights
        mask = (self.nearest_nodes[:,1:] == -1).unsqueeze(0).unsqueeze(1).unsqueeze(4).repeat(B, T, 1, 1, self.n_heads)
        weights = weights * (~mask)

        degree = weights.sum(3) # in (B, T, N, n_heads)
        degree_j = degree[:,:,self.nearest_nodes[:,1:].reshape(-1)].view(B, T, self.n_nodes, -1, self.n_heads) # in (B, T, N, k, n_heads)
        degree_multiply = torch.sqrt(degree.unsqueeze(3) * degree_j)
        inv_degree_multiply = torch.where(degree_multiply > 0, torch.ones((1,), device=self.device) / degree_multiply, torch.zeros((1,), device=self.device))
        inv_degree_multiply = torch.where(inv_degree_multiply == torch.inf, 0, inv_degree_multiply)
        weights = weights * inv_degree_multiply
        # print('undirected_weights', weights.shape)
        return weights # in (B, T, N, k, n_heads)
###############################TODO #####################################
    def directed_graph_from_features(self, features):
        '''
        Args:
            features (torch.Tensor) in (-1, T, n_nodes, n_features)
        Return:
            u_edges in (-1, T-1, n_edges, n_heads)
        '''
        B, T = features.size(0), features.size(1)
        weights = {}
        # pad features
        pad_features = torch.zeros_like(features[:,:,0], device=self.device).unsqueeze(2)
        pad_features = torch.cat((features, pad_features), dim=2)

        feature_i = pad_features[:,:-1, self.nearest_nodes.view(-1)].view(B, T-1, self.n_nodes, -1, self.n_heads, self.n_channels) # in (B, T-1, N, k, n_heads, n_channels)
        feature_j = features[:,1:] # in (B, T-1, N, n_heads, n_channels)
        Q_i = torch.einsum('hij, btnehj -> btnehi', self.multiQ1, feature_i)
        Q_j = torch.einsum('hij, btnhj -> btnhi', self.multiQ2, feature_j)
        # print('Qi,Qj', Q_i.shape, Q_j.shape)
        assert not torch.isnan(Q_j).any(), f'Q_j has NaN value: Q2 in ({self.multiQ2.max().item():.4f}, {self.multiQ2.min().item():.4f}; features in ({feature_j.max().item()}, {feature_j.min().item()}))'
        assert not torch.isnan(Q_i).any(), f'Q_i has NaN value: Q1 in ({self.multiQ1.max().item():.4f}, {self.multiQ1.min().item():.4f}, features in ({feature_i.max()}, {feature_i.min()})'
        weights = torch.exp(- (Q_i * Q_j.unsqueeze(3)).sum(-1)) # in (B, T-1, N, k, n_heads)
        # mask unused weights
        mask = (self.nearest_nodes == -1).unsqueeze(0).unsqueeze(1).unsqueeze(4).repeat(B, T-1, 1, 1, self.n_heads)
        weights = weights * (~mask)
        in_degree = weights.sum(3)
        # print('in_degree', in_degree.max(), in_degree.min(), torch.isnan(in_degree).any())
        inv_in_degree = torch.where(in_degree > 0, torch.ones((1,), device=self.device) / in_degree, torch.zeros((1,), device=self.device))
        inv_in_degree = torch.where(inv_in_degree == torch.inf, torch.zeros((1), device=self.device), inv_in_degree)
        # print('inv_in_degree', inv_in_degree.max(), inv_in_degree.min(), torch.isnan(inv_in_degree).any())
        weights = weights * inv_in_degree.unsqueeze(3)
        # print(weights.max(), weights.min(), torch.isnan(weights).any())
        return weights

    def forward(self, features=None):
        '''
        return u_ew and d_ew
        '''
        # print('features', features)
        assert features is not None, 'feature cannot be none'
        return self.undirected_graph_from_features(features), self.directed_graph_from_features(features)
        
# u_edges = torch.Tensor([[0,1], [1,0], [1,2], [2,1]]).type(torch.long)
# glm = GraphLearningModule(1, 3, u_edges, torch.Tensor([1,1,2,2]), initialize=True, device='cpu', n_heads=1)
# print(glm.undirected_graph_from_distance())