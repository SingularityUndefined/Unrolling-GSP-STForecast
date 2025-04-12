import torch
import torch.nn as nn
from lib.graph_learning_module_old1 import GraphLearningModule # GNNExtrapolation, FeatureExtractor, GraphLearningModule, GNNExtrapolation, GALExtrapolation
from lib.admm_block import ADMMBlock
from lib.backup_modules import layer_norm_on_data, layer_recovery_on_data, find_k_nearest_neighbors, SpatialTemporalEmbedding, LR_guess, connect_list
from torch.nn.parameter import Parameter
from lib.feature_extractor import GNNExtrapolation, GraphSAGEExtrapolation, FeatureExtractor

class UnrollingModel(nn.Module):
    def __init__(self, num_blocks, device, 
                 T, t_in,
                 n_heads,
                 interval,
                 signal_channels,
                 feature_channels,
                 k_hop, 
                 GNN_alpha=0.2,
                 # graph_sigma=6,
                 graph_info = {
                     'n_nodes': None,
                     'u_edges': None,
                     'u_dist': None,
                 },
                 ADMM_info = {
                 'ADMM_iters':30,
                 'CG_iters': 3,
                 'PGD_iters': 3,
                 'mu_u_init':3,
                 'mu_d1_init':3,
                 'mu_d2_init':3,
                 },
                 use_norm=False,
                 # use_dist_conv=False,
                 GNN_layers=2,
                 use_st_emb=True,
                 st_emb_info = {
                     'spatial_dim': 5,
                     't_dim': 10,
                     'tid_dim': 6,
                     'diw_dim': 4
                 },
                 # TODO: change here
                 use_extrapolation=True, # False for LR guess
                 use_old_extrapolation=False,
                 # use_LR_guess=False,
                 extrapolation_agg_layers=1,
                 sigma_ratio=450,
                 ablation='None',
                 use_one_channel=False,
                 shared_params=True,
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.device = device
        self.T = T
        self.t_in = t_in
        self.n_heads = n_heads
        self.use_norm = use_norm
        self.ablation = ablation
        self.use_one_channel = use_one_channel

        # define a graph connection pattern
        self.kNN = None
        self.nearsest_nodes, self.nearest_dists = find_k_nearest_neighbors(graph_info['n_nodes'], graph_info['u_edges'], graph_info['u_dist'], k_hop, device=self.device)
        self.connect_list = connect_list(graph_info['n_nodes'], graph_info['u_edges'], self.device)
        self.use_extrapolation = use_extrapolation
        if self.use_extrapolation:
            self.use_old_extrapolation = use_old_extrapolation
            if self.use_old_extrapolation:
                self.linear_extrapolation = GNNExtrapolation(graph_info['n_nodes'], t_in, T, self.nearsest_nodes, self.nearest_dists, n_heads, self.device, sigma_ratio=sigma_ratio)
            else:
                self.linear_extrapolation = GraphSAGEExtrapolation(graph_info['n_nodes'], t_in, T, self.nearsest_nodes, signal_channels, n_heads, device, interval=interval, n_layers=extrapolation_agg_layers)
                # self.linear_extrapolation = GALExtrapolation(graph_info['n_nodes'], t_in, T, self.nearsest_nodes, signal_channels, n_heads, device, n_layers=extrapolation_agg_layers)
        
        self.use_st_emb = use_st_emb
        if self.use_st_emb:
            self.st_emb = SpatialTemporalEmbedding(graph_info['n_nodes'], graph_info['u_edges'], graph_info['u_dist'], sigma_ratio, self.device, st_emb_info['spatial_dim'], st_emb_info['t_dim'], st_emb_info['tid_dim'], st_emb_info['diw_dim'])
            signal_emb_channels = signal_channels + st_emb_info['spatial_dim'] + st_emb_info['t_dim'] + st_emb_info['tid_dim'] + st_emb_info['diw_dim']
        else:
            signal_emb_channels = signal_channels
        
        if self.use_one_channel:
            signal_rec_channels = 1
            signal_rec_emb_channels = signal_emb_channels - signal_channels + 1
        else:
            signal_rec_channels = signal_channels
            signal_rec_emb_channels = signal_emb_channels

        print('signal_channels', signal_channels)
        print('signal_emb_channels', signal_emb_channels)
        print('signal_rec_channels', signal_rec_channels)
        print('signal_rec_emb_channels', signal_rec_emb_channels)
        self.model_blocks = nn.ModuleList([])

        self.skip_connection_weights = Parameter(torch.ones((num_blocks,), device=self.device) * 0.95, requires_grad=True)

        # spatiotemporal embeddings
        

        for i in range(self.num_blocks):
            self.model_blocks.append(nn.ModuleDict(
                {
                    'feature_extractor':FeatureExtractor(
                        in_features=signal_emb_channels if i == 0 else signal_rec_emb_channels,
                        out_features=feature_channels,
                        nearest_nodes=self.nearsest_nodes,
                        n_heads=n_heads,
                        device=device,
                        interval=interval
                    ),
                    # 'feature_extractor': FeatureExtractor(
                    #     n_in=signal_emb_channels if i == 0 else signal_rec_emb_channels,
                    #     n_out=feature_channels,
                    #     # n_nodes=graph_info['n_nodes'],
                    #     n_heads=n_heads,
                    #     nearest_nodes=self.nearsest_nodes,
                    #     # nearest_dists=self.nearest_dists,
                    #     device=device,
                    #     n_layers=GNN_layers,
                    #     # sigma=graph_sigma,
                    #     alpha=GNN_alpha,
                    #     # use_dist_conv=use_dist_conv,
                    #     use_graph_agg=True,
                    #     n_nodes=graph_info['n_nodes'],
                    #     sigma_ratio=450,
                    #     nearest_dist=self.nearest_dists
                    # ),
                    'ADMM_block': ADMMBlock(
                        T=T,
                        n_nodes=graph_info['n_nodes'],
                        n_heads=n_heads,
                        n_channels=signal_rec_channels,
                        interval=interval,
                        connect_list=self.connect_list,
                        nearest_nodes=self.nearsest_nodes,
                        device=device,
                        ADMM_info=ADMM_info,
                        ablation=self.ablation
                    ),
                    'graph_learning_module': GraphLearningModule(
                        T=T,
                        n_nodes=graph_info['n_nodes'],
                        connect_list=self.connect_list,
                        nearest_nodes=self.nearsest_nodes,
                        n_heads=n_heads,
                        interval=interval,
                        device=device,
                        n_channels=feature_channels,
                        shared_params=shared_params,
                    )
                }
            ))
        self.y_norm_shape = [self.t_in, graph_info['n_nodes'], signal_channels]
        self.norm_shape = [self.T, graph_info['n_nodes'], signal_channels]
    
    def regularized_terms(self, x, t=None): # TODO: save as the same operators
        '''
        Notice that for here, x stands for the true sequence, the full observation, the ideal x. Just for testing and validation.
    Inputs:
        u_model: unrolling models (T)
        x (torch.Tensor): in (B, T, n_nodes, n_channels)
    Return:
        regularized term (x L^u x, ||Ld x||_2, ||L_d x||_1)
    '''
        assert not self.training, 'only on validation and test'
        B = x.size(0)
        if self.use_norm:
            x, mean, std = layer_norm_on_data(x, self.norm_shape)

        x_norm_list = []
        x_Lu_norm_list = []
        Ldx_l2_list = []
        Ldx_l1_list = []
        with torch.no_grad():
            for i in range(self.num_blocks):
                t_block = self.model_blocks[i]

                feature_extractor = t_block['feature_extractor']
                graph_learn = t_block['graph_learning_module']
                admm_block = t_block['ADMM_block']
                # mus
                # mu_u = admm_block.mu_u.max()
                # mu_d1 = admm_block.mu_d1.max()
                # mu_d2 = admm_block.mu_d2.max()
                # passing forward
                # features, regenerate from original signal
                features = feature_extractor(x, self.T, self.GNN_graph) # in (batch, T, n_nodes, n_heads, n_out)
            
                u_ew, d_ew = graph_learn(features)

                admm_block.u_ew = u_ew
                admm_block.d_ew = d_ew

                # cauculate norms, x in (B, T, n_nodes, 1)

                # pass on the module, caucluate norms
                p = self.skip_connection_weights[i]

                x_norm_list.append(torch.norm(x, dim=0))
                Lu_norm = torch.sqrt((x * admm_block.apply_op_Lu(x)).sum([1,2,3])) # L2 norm
                x_Lu_norm_list.append(Lu_norm)
                Ld_l2 = torch.norm(admm_block.apply_op_Ldr(x))
                Ld_l1 = torch.norm(admm_block.apply_op_Ldr(x), p=1)
                Ldx_l2_list.append(Ld_l2)
                Ldx_l2_list.append(Ld_l1)
                # pass on
                x_old = x
                x_new = admm_block(x, self.t_in)
                x = p * x_new + (1-p) * x_old
                
            # # calculate mean of norms0
            # x_norm = torch.cat(x_norm_list, dim=0).mean()
            # Lu_norm = torch.cat(x_Lu_norm_list, dim=0).mean()
            # Ldx_l1 = torch.cat(Ldx_l1_list, dim=0).mean()
            # Ldx_l2 = torch.cat(Ldx_l2_list, dim=0).mean()
        # organize the feature dicts
        
        return torch.Tensor(x_norm_list), torch.Tensor(x_Lu_norm_list), torch.Tensor(Ldx_l1_list), torch.Tensor(Ldx_l2_list) # in (n_blocks, B)
    
    def clamp_param(self, alpha_max=None, beta_max=None):
        for i in range(self.num_blocks):
            transformer_block = self.model_blocks[i]
            graph_learn: GraphLearningModule = transformer_block['graph_learning_module']
            admm_block: ADMMBlock = transformer_block['ADMM_block']

            if alpha_max is not None:
                admm_block.alpha_x.data = torch.clamp(admm_block.alpha_x.data, 0.0, alpha_max)
                admm_block.alpha_zu.data = torch.clamp(admm_block.alpha_zu.data, 0.0, alpha_max)
                if self.ablation != 'DGLR':
                    admm_block.alpha_zd.data = torch.clamp(admm_block.alpha_zd.data, 0.0, alpha_max)
            else:
                admm_block.alpha_x.data = torch.clamp(admm_block.alpha_x.data, 0.0)
                admm_block.alpha_zu.data = torch.clamp(admm_block.alpha_zu.data, 0.0)
                if self.ablation != 'DGLR':
                    admm_block.alpha_zd.data = torch.clamp(admm_block.alpha_zd.data, 0.0)

            if beta_max is not None:
                admm_block.beta_x.data = torch.clamp(admm_block.beta_x.data, 0.0, beta_max)
                admm_block.beta_zu.data = torch.clamp(admm_block.beta_zu.data, 0.0, beta_max)
                if self.ablation != 'DGLR':
                    admm_block.beta_zd.data = torch.clamp(admm_block.beta_zd.data, 0.0, beta_max)
            else:
                admm_block.beta_x.data = torch.clamp(admm_block.beta_x.data, 0.0)
                admm_block.beta_zu.data = torch.clamp(admm_block.beta_zu.data, 0.0)
                if self.ablation != 'DGLR':
                    admm_block.beta_zd.data = torch.clamp(admm_block.beta_zd.data, 0.0)


    def forward(self, y, t_list):
        '''
        y in (batch, t, n_nodes, signal_channels)
        '''
        # linear extrapolation
        # print('y', y.size(), 't_list', t_list.size())
        B, t, signal_channels = y.size(0), y.size(1), y.size(-1)
        if self.use_norm:
            y, mean, std = layer_norm_on_data(y, self.y_norm_shape)

        # padding zeros
        # pad = torch.zeros((B, t, 1, signal_channels), device=self.device)  
        # y = torch.cat((y, pad), dim=2)
        # print('pad y', y.shape, y[:,:,-1].sum())
        if self.use_extrapolation:
            output = self.linear_extrapolation(y)
        else:
            output = LR_guess(y, self.T, self.device)

        assert not torch.isnan(output).any(), 'linear extrapolation has nan'
        # print('pad output', output.size(), output[:,:,-1].sum())
        if self.use_st_emb:
            shared_output_emb = self.st_emb(t_list)

        for i in range(self.num_blocks):
            # print('block', i)
            if self.use_st_emb:
            # print('self.st_emb.device', self.st_emb.device)
                output_emb = torch.cat((output, shared_output_emb), -1)# self.st_emb(output, t_list)
            else:
                output_emb = output
            # print('output_emb', output_emb.size())
            if self.use_one_channel and i == 0:
                output_old = output[...,0:1]
            else:
                output_old = output

            transformer_block = self.model_blocks[i]
            feature_extractor = transformer_block['feature_extractor']
            graph_learn = transformer_block['graph_learning_module']
            admm_block = transformer_block['ADMM_block']
            # learn features
            try:
                features = feature_extractor(output_emb) # in (batch, T, n_nodes, n_heads, n_out)
            except ValueError as ae:
                raise ValueError(f'Error in Feature extractor in Block {i}: {ae}') from ae
            # print('features', features.size())
            try:
                u_ew, d_ew = graph_learn(features)
            except AssertionError as ae:
                raise ValueError(f'Error in Graph Learning Module in Block {i}: {ae}') from ae
            # print('max in weights', get_max_in_dict(u_ew), get_max_in_dict(d_ew))
            # print('u_ew', u_ew[0].shape)
            # print('d_ew', d_ew[0].shape)
            admm_block.u_ew = u_ew
            admm_block.d_ew = d_ew
            try:
                if self.use_one_channel:
                    output_new = admm_block(output[...,0:1], t)
                else:
                    output_new = admm_block(output, t)
                # print('output_new', output_new.size())
                # output_new = admm_block(output, t) # in (batch, T, n_nodes, signal_channels)
            # skip connections
            except AssertionError as ae:
                raise ValueError(f'Assertation Error in ADMM block in block {i} - {ae}') from ae
            assert not torch.isnan(output_new).any(), f'output_new has NaN value in block {i}'
            p = self.skip_connection_weights[i]
            assert not torch.isnan(self.skip_connection_weights).any(), f'skip connection has NaN Values in block {i}'
            output = p * output_new + (1-p) * output_old
            # print(f'output after block {i}: {output.size()}')

        if self.use_norm:
            output = layer_recovery_on_data(output, self.norm_shape, mean, std)

        # print('output', output.size())
        return output        # 

def get_max_in_dict(ew:dict):
    maxlist = []
    for v in ew.values():
        maxlist.append(v.max())
    return max(maxlist)
