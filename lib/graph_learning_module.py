import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
# from backup_modules import connect_list


class GraphLearningModule(nn.Module):
    '''
    learning the directed and undirected weights from features
    '''
    def __init__(self, T, n_nodes, connect_list, nearest_nodes, n_heads, interval, device, n_channels=None, sigma=6, Q1_init=1.2, M_init=2, sharedM=True, sharedQ=True, diff_interval=True, directed_time=True) -> None:
        '''
        Args:
            u_edges (torch.Tensor) in (n_edges, 2) # nodes regularized
            u_dist (torch.Tensor) in (n_edges)
        We construct d_edges by hand with n_nodes
        '''
        super().__init__()
        self.directed_time = directed_time
        self.T = T
        self.n_nodes = n_nodes
        self.device = device
        # construct d_edges, d_dist
        self.connect_list = connect_list #(N, k)
        self.nearest_nodes = nearest_nodes
        # multi_heads
        self.n_heads = n_heads
        self.interval = interval
        # self.temp_indice = torch.arange(0, T).reshape(-1, 1) - torch.arange(1, interval + 1) # in (T, interval)
        self.temp_indice = torch.arange(1, T).reshape(-1, 1) - torch.arange(1, interval + 1) # in (T - 1, interval)

        # self.n_features = n_features # feature channels
        self.n_channels = n_channels
        self.n_out = (self.n_channels + 1) // 2
        # define multiM, multiQs
        self.sharedM = sharedM
        self.sharedQ = sharedQ
        self.diff_interval = diff_interval
        self.Q1_init = Q1_init
        # self.Q2_init = Q2_init
        self.M_init = M_init

        q_form = torch.diag_embed(torch.ones((self.n_heads, self.n_channels), device=self.device))
        # q_form = torch.zeros((self.n_heads, self.n_channels, self.n_channels), device=self.device)
        # q_form[:,:self.n_out, :self.n_out] = torch.diag_embed(torch.ones((self.n_heads, self.n_out), device=self.device)) # tall matrix, self.n_out > self.n_channels

        # Question: low-rank assumption, makes sense?
        # add random noise, small value
        # q_form = q_form + torch.randn((self.n_heads, self.n_channels, self.n_channels), device=self.device) * 0.01
        # all variables shared across time
        multiQ_init = q_form # * self.Q1_init
        # multiQ2_init = q_form * self.Q2_init
        multiM_init = torch.diag_embed(torch.ones((self.n_heads, self.n_channels), device=self.device)) * self.M_init

        if not self.sharedQ:
            # multiQ1_init = multiQ1_init.unsqueeze(0).repeat(T-1, 1, 1, 1)
            # multiQ2_init = multiQ2_init.unsqueeze(0).repeat(T-1, 1, 1, 1)
            multiQ_init = multiQ_init.unsqueeze(0).repeat(T-1, 1, 1, 1)
        if not self.sharedM:
            # multiQ2_init = multiQ2_init.unsqueeze(0).repeat(T, 1, 1, 1)
            multiM_init = multiM_init.unsqueeze(0).repeat(T, 1, 1 , 1)
        
        if self.diff_interval:
            if self.sharedQ:
                multiQ_init = multiQ_init.unsqueeze(0).repeat(self.interval, 1, 1, 1) # in (interval, n_heads, n_channels, n_channels)
                multiQ_init = multiQ_init * torch.linspace(1, Q1_init, steps=self.interval).reshape(self.interval, 1, 1, 1).to(self.device) # in (interval, n_heads, n_channels, n_channels)
            else:
                multiQ_init = multiQ_init.unsqueeze(1).repeat(1, self.interval, 1, 1, 1) # in (T-1, interval, n_heads, n_channels, n_channels)
                multiQ_init = multiQ_init * torch.linspace(1, Q1_init, steps=self.interval).reshape(1, self.interval, 1, 1, 1).to(self.device) # in (T-1, interval, n_heads, n_channels, n_channels)

        self.multiQ = Parameter(multiQ_init, requires_grad=True)
        # self.multiQ2 = Parameter(multiQ2_init, requires_grad=True)
        self.multiM = Parameter(multiM_init, requires_grad=True) # in (n_heads, n_channels, n_channels)

# ####################################### KNN VERSION #######################################
    def undirected_graph_from_features(self, features):
        '''
        Args:
            features (torch.Tensor) in (-1, T, n_nodes, n_heads, n_channels)
        Returns:
            u_edges in (-1, T, n_edges, n_heads)
        '''
        B, T = features.size(0), features.size(1)

        # pad features
        # nn = self.nearest_nodes[:, 1:]
        pad_features = torch.zeros_like(features[:,:,0], device=self.device).unsqueeze(2)
        pad_features = torch.cat((features, pad_features), dim=2)

        feature_j = pad_features[:,:,self.nearest_nodes[:,1:].reshape(-1)].view(B, T, self.n_nodes, -1, self.n_heads, self.n_channels)
        # print(features.size(), feature_j.size())

        df = features.unsqueeze(3) - feature_j # in (B, T, N, k, n_heads, n_channels)
        # print(self.multiM.size(), df.size())
        if self.sharedM:
            Mdf = torch.einsum('hij, btnehj -> btnehi', self.multiM, df)
        else:
            Mdf = torch.einsum('thij, btnehj -> btnehi', self.multiM, df) # in (B, T, N, k, n_heads, n_channels)
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


################################## PHYSICAL GRAPH VERSION #################################################
    # def undirected_graph_from_features(self, features):
    #     '''
    #     Args:
    #         features (torch.Tensor) in (-1, T, n_nodes, n_heads, n_channels)
    #     Returns:
    #         u_edges in (-1, T, n_edges, n_heads)
    #     '''
    #     B, T = features.size(0), features.size(1)
    #     weights = {}

    #     # pad features
    #     # nn = self.nearest_nodes[:, 1:]
    #     pad_features = torch.zeros_like(features[:,:,0], device=self.device).unsqueeze(2)
    #     pad_features = torch.cat((features, pad_features), dim=2) # in (B, T, N+1, n_heads, n_channels)

    #     feature_j = pad_features[:,:,self.connect_list[:,1:].reshape(-1)].view(B, T, self.n_nodes, -1, self.n_heads, self.n_channels)
    #     # print(features.size(), feature_j.size())

    #     df = features.unsqueeze(3) - feature_j # in (B, T, N, k, n_heads, n_channels)
    #     if self.shared_params:
    #         Mdf = torch.einsum('hij, btnehj -> btnehi', self.multiM, df) # in (B, T, N, k, n_heads, n_channels)
    #     else:
    #         Mdf = torch.einsum('thij, btnehj -> btnehi', self.multiM, df)

    #     weights = torch.exp(- (Mdf ** 2).sum(-1)) # in (B, T, N, k, n_heads)
    #     # mask weights
    #     mask = (self.connect_list[:,1:] == -1).unsqueeze(0).unsqueeze(1).unsqueeze(4).repeat(B, T, 1, 1, self.n_heads)
    #     weights = weights * (~mask) # if mask true, weights = 0

    #     degree = weights.sum(3) # in (B, T, N, n_heads)
    #     degree_j = degree[:,:,self.connect_list[:, 1:].reshape(-1)].view(B, T, self.n_nodes, -1, self.n_heads) # in (B, T, N, k, n_heads)
    #     degree_multiply = torch.sqrt(degree.unsqueeze(3) * degree_j)

    #     inv_degree_multiply = torch.where(degree_multiply > 0, torch.ones((1,), device=self.device) / degree_multiply, torch.zeros((1,), device=self.device))
    #     inv_degree_multiply = torch.where(inv_degree_multiply == torch.inf, 0, inv_degree_multiply)
    #     weights = weights * inv_degree_multiply
    #     # print('undirected_weights', weights.shape)
    #     return weights # in (B, T, N, k, n_heads)
###############################TODO #####################################
################################Physical Graph Version##############################################
    # def directed_graph_from_features(self, features):
    #     '''
    #     Args:
    #         features (torch.Tensor) in (-1, T, n_nodes, n_features)
    #     Return:
    #         u_edges in (-1, T-1, n_edges, n_heads)
    #     '''
    #     B, T = features.size(0), features.size(1)
    #     weights = {}
    #     # pad features
    #     pad_features = torch.zeros_like(features[:,:,0], device=self.device).unsqueeze(2)
    #     pad_features = torch.cat((features, pad_features), dim=2)

    #     feature_i = pad_features[:,:-1, self.connect_list.view(-1)].view(B, T-1, self.n_nodes, -1, self.n_heads, self.n_channels) # in (B, T-1, N, k, n_heads, n_channels)
    #     feature_j = features[:,1:] # in (B, T-1, N, n_heads, n_channels)
    #     if self.shared_params:
    #         Q_i = torch.einsum('hij, btnehj -> btnehi', self.multiQ1, feature_i)
    #         Q_j = torch.einsum('hij, btnhj -> btnhi', self.multiQ2, feature_j)
    #     else:
    #         Q_i = torch.einsum('thij, btnehj -> btnehi', self.multiQ1, feature_i)
    #         Q_j = torch.einsum('thij, btnhj -> btnhi', self.multiQ2, feature_j)

    #     # print('Qi,Qj', Q_i.shape, Q_j.shape)
    #     assert not torch.isnan(Q_j).any(), f'Q_j has NaN value: Q2 in ({self.multiQ2.max().item():.4f}, {self.multiQ2.min().item():.4f}; features in ({feature_j.max().item()}, {feature_j.min().item()}))'
    #     assert not torch.isnan(Q_i).any(), f'Q_i has NaN value: Q1 in ({self.multiQ1.max().item():.4f}, {self.multiQ1.min().item():.4f}, features in ({feature_i.max()}, {feature_i.min()})'
    #     weights = torch.exp(- (Q_i * Q_j.unsqueeze(3)).sum(-1)) # in (B, T-1, N, k, n_heads)
    #     # mask unused weights
    #     mask = (self.connect_list == -1).unsqueeze(0).unsqueeze(1).unsqueeze(4).repeat(B, T-1, 1, 1, self.n_heads)
    #     weights = weights * (~mask)
    #     in_degree = weights.sum(3)
    #     # print('in_degree', in_degree.max(), in_degree.min(), torch.isnan(in_degree).any())
    #     inv_in_degree = torch.where(in_degree > 0, torch.ones((1,), device=self.device) / in_degree, torch.zeros((1,), device=self.device))
    #     inv_in_degree = torch.where(inv_in_degree == torch.inf, torch.zeros((1), device=self.device), inv_in_degree)
    #     # print('inv_in_degree', inv_in_degree.max(), inv_in_degree.min(), torch.isnan(inv_in_degree).any())
    #     weights = weights * inv_in_degree.unsqueeze(3)
    #     # print(weights.max(), weights.min(), torch.isnan(weights).any())
    #     return weights
 ############################# Dense connection line graph version ##############################################
    def directed_graph_from_features(self, features):
        '''
        Args:
            features (torch.Tensor) in (-1, T, n_nodes, n_heads, n_channels)
        Return:
            u_edges in (-1, T, self.interval, n_nodes, n_heads, n_channels), with a lower triangular mask
        '''
        # print('features', features.shape, features.max(), features.min(), torch.isnan(features).any())
        B, T, C = features.size(0), features.size(1), features.size(-1)
        features_j = features[:,1:] # in (B, T-1, N, n_heads, n_channels)
        # father features
        # father_features = features# .unsqueeze(2) # in (B, T, 1, n_nodes, n_heads, n_channels)
        # children features
        features_i = features[:,self.temp_indice.view(-1)].view(B, T-1, self.interval, self.n_nodes, -1, self.n_channels) # in (B, T - 1, interval, N, n_heads, n_channels)
        df = features_i - features_j.unsqueeze(2)
        # multiply with Qs
        if self.sharedQ:
            if self.diff_interval:
                Q_df = torch.einsum('vhij, btvnhj -> btvnhi', self.multiQ, df) # in (B, T-1, interval, N, n_heads, n_channels)
            # Q_j = torch.einsum('hij, btnhj -> btnhi', self.multiQ, features_j)
            else:
                Q_df = torch.einsum('hij, btvnhj -> btvnhi', self.multiQ, df)
        else:
            if self.diff_interval:
                # Q_j = torch.einsum('thij, btnhj -> btnhi', self.multiQ, features_j)
                Q_df = torch.einsum('tvhij, btvnhj -> btvnhi', self.multiQ, df)
            # Q_j = torch.einsum('thij, btnhj -> btnhi', self.multiQ, features_j)
            else:
                Q_df = torch.einsum('thij, btvnhj -> btvnhi', self.multiQ, df)
        # assertation
        assert not torch.isnan(Q_df).any(), f'Q_j has NaN value: Q in ({self.multiQ.max().item():.4f}, {self.multiQ.min().item():.4f}; features in ({features.max().item()}, {features.min().item()}))'
        # assert not torch.isnan(Q_i).any(), f'Q_i has NaN value: Q1 in ({self.multiQ1.max().item():.4f}, {self.multiQ1.min().item():.4f}, features in ({features.max()}, {features.min()})'
        # multiply two qs
        d = Q_df ** 2 # Q_df * df # in (B, T, interval, N, n_heads, n_channels)
        # print('deplacement', d.max(), d.min(), torch.isnan(d).any())
        weights = torch.exp(-d).sum(-1) # in (B, T, interval, N, n_heads)

        # mask
        mask = torch.ones(T-1, self.interval).tril_(diagonal=0).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(B, 1, 1, self.n_nodes, self.n_heads).to(self.device)
        # mask = torch.ones(T, self.interval).tril_(diagonal=-1).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(B, 1, 1, self.n_nodes, self.n_heads).to(self.device) # in (B, T-1, interval, N, n_heads)
        weights = weights * mask
        # print('weights before normalization', weights.shape, weights[mask_bool].max(), weights[mask_bool].min(), torch.isnan(weights).any())

        # # normalization
        if self.directed_time:
            in_degree = weights.sum(2, keepdim=True) # in (B, T-1, interval, N, n_heads)
            inv_in_degree = torch.where(in_degree > 0, torch.ones((1,), device=self.device) / in_degree, torch.zeros((1,), device=self.device))
            weights = weights * inv_in_degree # in (B, T, interval, N, n_head)
        
        else:
            # normalize according to  undirected graph
            in_degree = weights.sum(2) # in (B, T-1, N, n_heads)
            out_degree = torch.stack([weights.diagonal(offset=-offset, dim1=1, dim2=2).sum(-1) for offset in range(T-1)], dim=1) # in (B, T-1, N, n_heads)
            pad_degree = torch.zeros_like(in_degree[:,0:1], device=self.device) # in (B, 1, N, n_heads)
            degree = torch.cat((pad_degree, in_degree), dim=1) + torch.cat((out_degree, pad_degree), dim=1) # in (B, T, N, n_heads)

            degree_i = degree[:,1:].unsqueeze(2) # in (B, T-1, 1, N, n_heads)
            degree_j = degree[:,self.temp_indice.view(-1)].view(B, T-1, self.interval, self.n_nodes, self.n_heads) # in (B, T-1, interval, N, n_heads)
            # print(degree_i.shape, degree_j.shape)
            degree_multiply = degree_i * degree_j # in (B, T-1, interval, N, n_heads)
            inv_degree_multiply = torch.where(degree_multiply > 0, torch.ones((1,), device=self.device) / degree_multiply, torch.zeros((1,), device=self.device))
            weights = weights * torch.sqrt(inv_degree_multiply)


            # degree = torch.cat((degree, pad_degree), dim=1) # in (B, T, N, n_heads)

        # or
       #  weights = weights / (weights.sum(2, keepdim=True) + 1e-8) # in (B, T, interval, N, n_heads)

        # print weights
        # print('weights', weights.max(), weights.min(), torch.isnan(weights).any())
        # print('weights', weights.shape, weights.max(), weights.min(), torch.isnan(weights).any())
        # mask_bool = mask.to(torch.bool)
        # print('weights', weights[mask_bool].max(), weights[mask_bool].min(), torch.isnan(weights).any())
        return weights

######################### kNN version #################################
    # def directed_graph_from_features(self, features):
    #     '''
    #     Args:
    #         features (torch.Tensor) in (-1, T, n_nodes, n_features)
    #     Return:
    #         u_edges in (-1, T-1, n_edges, n_heads)
    #     '''
    #     B, T = features.size(0), features.size(1)
    #     weights = {}
    #     # pad features
    #     pad_features = torch.zeros_like(features[:,:,0], device=self.device).unsqueeze(2)
    #     pad_features = torch.cat((features, pad_features), dim=2)

    #     feature_i = pad_features[:,:-1, self.nearest_nodes.view(-1)].view(B, T-1, self.n_nodes, -1, self.n_heads, self.n_channels) # in (B, T-1, N, k, n_heads, n_channels)
    #     feature_j = features[:,1:] # in (B, T-1, N, n_heads, n_channels)
    #     if self.shared_params:
    #         Q_i = torch.einsum('hij, btnehj -> btnehi', self.multiQ1, feature_i)
    #         Q_j = torch.einsum('hij, btnhj -> btnhi', self.multiQ2, feature_j)
    #     else:
    #         Q_i = torch.einsum('thij, btnehj -> btnehi', self.multiQ1, feature_i)
    #         Q_j = torch.einsum('thij, btnhj -> btnhi', self.multiQ2, feature_j)

    #     # print('Qi,Qj', Q_i.shape, Q_j.shape)
    #     assert not torch.isnan(Q_j).any(), f'Q_j has NaN value: Q2 in ({self.multiQ2.max().item():.4f}, {self.multiQ2.min().item():.4f}; features in ({feature_j.max().item()}, {feature_j.min().item()}))'
    #     assert not torch.isnan(Q_i).any(), f'Q_i has NaN value: Q1 in ({self.multiQ1.max().item():.4f}, {self.multiQ1.min().item():.4f}, features in ({feature_i.max()}, {feature_i.min()})'
    #     weights = torch.exp(- (Q_i * Q_j.unsqueeze(3)).sum(-1)) # in (B, T-1, N, k, n_heads)
    #     # mask unused weights
    #     mask = (self.nearest_nodes == -1).unsqueeze(0).unsqueeze(1).unsqueeze(4).repeat(B, T-1, 1, 1, self.n_heads)
    #     weights = weights * (~mask)
    #     in_degree = weights.sum(3)
    #     # print('in_degree', in_degree.max(), in_degree.min(), torch.isnan(in_degree).any())
    #     inv_in_degree = torch.where(in_degree > 0, torch.ones((1,), device=self.device) / in_degree, torch.zeros((1,), device=self.device))
    #     inv_in_degree = torch.where(inv_in_degree == torch.inf, torch.zeros((1), device=self.device), inv_in_degree)
    #     # print('inv_in_degree', inv_in_degree.max(), inv_in_degree.min(), torch.isnan(inv_in_degree).any())
    #     weights = weights * inv_in_degree.unsqueeze(3)
    #     # print(weights.max(), weights.min(), torch.isnan(weights).any())
    #     return weights

####################################
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