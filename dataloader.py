# loading data for traffic flow forecasting
# testing METR-LA
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from collections import Counter
import pickle


# data_folder = os.path.join('UnrollingForecasting/MainExperiments/datasets/data/', 'PEMS03')
# graph_csv = 'PEMS03.csv'
# data_file = 'PEMS03.npz'
# id_file = 'PEMS03.txt'

# sensor_id = np.loadtxt(os.path.join(data_folder, id_file), dtype=int)
# # print(sensor_id.shape)
# n_vertex = sensor_id.shape[0]
# sensor_dict = dict([(sensor_id[k], k) for k in range(n_vertex)])
# print(sensor_dict)

# data = np.load(os.path.join(data_folder, data_file))['data']
# print(data.shape)

# df = pd.read_csv(os.path.join(data_folder, graph_csv), error_bad_lines=False, index_col=None)
# df.info()
# print(df.head(5))
# print(df['from'].values)
# from_list = [sensor_dict[i] for i in df['from'].values]
# to_list = [sensor_dict[i] for i in df['to'].values]
# n_edges = len(from_list)
# u_edges = np.array([from_list + to_list, to_list + from_list]).T
# from collections import Counter
# dic = Counter([(u_edges[i,0], u_edges[i,1]) for i in range(n_edges)])
# assert max(list(dic.values())), 'distance graph asymmetric'
# print(u_edges)
# ew1 = df['distance'].values
# u_distance = np.stack([ew1, ew1]).reshape(-1)


def physical_graph(df, sensor_dict=None):
    if sensor_dict is None:
        from_list, to_list = list(df['from'].values), list(df['to'].values)
    else:
        from_list = [sensor_dict[i] for i in df['from'].values]
        to_list = [sensor_dict[i] for i in df['to'].values]
    n_edges = len(from_list) * 2
    # bi-directional
    u_edges = np.array([from_list + to_list, to_list + from_list]).T
    dic = Counter([(u_edges[i,0], u_edges[i,1]) for i in range(n_edges)])
    assert max(list(dic.values())), 'distance graph asymmetric'
    ew1 = df[df.columns[-1]].values
    u_distance = np.stack([ew1, ew1]).reshape(-1)
    return n_edges, u_edges, u_distance




class TrafficDataset(Dataset):
    def __init__(self, data_folder, graph_csv, data_file, T, t, stride, split='train', n_nodes=None, id_file=None, return_time=False, use_one_channel=False, truncated=False): # , use_one_channel=False) -> None:
        '''
        train:val:test = 6:2:2
        Components:
            data: in (T_total, n_nodes, 1)
            df: table ['from', 'to', 'distance']
            n_nodes: int, number of nodes
            n_edges: int, number of edges (bi-directional)
            u_edges: in (n_edges, 2)
            u_distance: in (n_edges)
        '''
        super().__init__()
        # time series
        self.T = T
        self.t = t
        self.stride = stride
        self.truncated = truncated
        self.return_time = return_time
        data = np.load(os.path.join(data_folder, data_file))['data'] # (T, n_in)
        self.use_one_channel = use_one_channel
        # if self.use_one_channel:
        #     data = data[..., 0:1]
        
        print('nan_count', len(data[np.isnan(data)]))
        # print('datashape', data.shape, data[0:2])
        self.signal_channel = data.shape[-1]
        data_len = data.shape[0]
        if truncated:
            data_len = data_len // self.stride
            self.stride = 1
            
        # print('dat_len', data_len)
        assert split in ['train', 'val', 'test'], 'split should in train, val or test'
        if split == 'train':
            self.data_begin = 0
            self.data = data[0:int(data_len * 0.6)]
        elif split == 'val':
            self.data_begin = int(data_len * 0.6)
            self.data = data[int(data_len * 0.6):int(data_len * 0.8)]
        elif split == 'test':
            self.data_begin = int(data_len * 0.8)
            self.data = data[int(data_len * 0.8):]
        # graph
        self.df = pd.read_csv(os.path.join(data_folder, graph_csv),index_col=None)
        if id_file is not None:
            sensor_id = np.loadtxt(os.path.join(data_folder, id_file), dtype=int)
            self.n_nodes = sensor_id.shape[0]
            self.sensor_dict = dict([(sensor_id[k], k) for k in range(self.n_nodes)])
        else:
            self.sensor_dict = None
            self.n_nodes = max(max(self.df['from'].values), max(self.df['to'].values)) + 1
        self.n_edges, self.u_edges, self.u_distance = physical_graph(self.df, self.sensor_dict)
        self.u_edges = torch.Tensor(self.u_edges).type(torch.long)#, dtype=torch.long)
        self.u_distance = torch.Tensor(self.u_distance)
        self.d_edges = torch.cat([self.u_edges, torch.arange(0, self.n_nodes)[:,None] + torch.zeros((2,), dtype=torch.long)], 0)
        self.graph_info = {
            'n_nodes': self.n_nodes,
            'u_edges': self.u_edges,
            'u_dist': self.u_distance
        }
        # print(self.d_edges)

    def __len__(self):
        return (self.data.shape[0] - self.T) // self.stride
    
    def __getitem__(self, index):
        y = self.data[index * self.stride:index * self.stride + self.t] # in (t, n_nodes, 1)
        x = self.data[index * self.stride:index * self.stride + self.T] # in (T, n_nodes, 1)
        # model(y) = x
        if self.use_one_channel:
            x = x[...,0:1]
        time = torch.arange(0, self.T).type(torch.long) + index * self.stride + self.data_begin
        if self.return_time:
            return torch.Tensor(y), torch.Tensor(x), time
        else:
            return torch.Tensor(y), torch.Tensor(x)


def directed_physical_graph(adj_mat, squared_dist=False):
    u_edges = []
    u_distance = []
    print('original edges', (adj_mat > 0).sum())
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if i != j and adj_mat[i, j] > 0: 
                u_edges.append([i, j])
                if squared_dist: 
                    u_distance.append(np.sqrt(-np.log(adj_mat[i, j])))
                else:
                    u_distance.append(-np.log(adj_mat[i, j]))
    
    print('uedges original', len(u_distance))
    for i in range(adj_mat.shape[0]):
        adj_mat[i, i] = 0
    # isolated nodes
    n_isolated = 0
    for i in range(adj_mat.shape[0]):
        if np.sum(adj_mat[i]) == 0:
            n_isolated += 1
            if i != 0:
                u_edges.append([i, i-1])
                u_edges.append([i-1, i])
                u_distance.append(1)
                u_distance.append(1)
            if i != adj_mat.shape[0] - 1:
                u_edges.append([i, i+1])
                u_edges.append([i+1, i])
                u_distance.append(1)
                u_distance.append(1)
    print("n_isolated", n_isolated)

    n_edges = len(u_edges)
    u_edges = np.array(u_edges)
    u_distance = np.array(u_distance)
    return n_edges, u_edges, u_distance

class DirectedTrafficDataset(Dataset):
    def __init__(self, data_folder, adj_mat_file, data_file, T, t, stride, split='train', n_nodes=None, return_time=False, use_one_channel=False) -> None:
        '''
        train:val:test = 6:2:2
        Components:
            data: in (T_total, n_nodes, n_channels)
            adj_mat: in (n_nodes, n_nodes)
            n_nodes: int, number of nodes
            n_edges: int, number of edges (directed)
            u_edges: in (n_edges, 2)
            u_distance: in (n_edges)
        '''
        super().__init__()
        # time series
        self.T = T
        self.t = t
        self.stride = stride
        self.return_time = return_time
        data = np.load(os.path.join(data_folder, data_file)) # (T, n_in)
        self.use_one_channel = use_one_channel
        # data = np.expand_dims(data[:, :, 0], axis=-1)

        # normalization
        # mean, std = data.mean(), data.std()
        # data = (data - mean) / std
        
        print('nan_count', len(data[np.isnan(data)]))
        # print('datashape', data.shape, data[0:2])
        self.signal_channel = data.shape[-1]
        data_len = data.shape[0]
        # print('dat_len', data_len)
        assert split in ['train', 'val', 'test'], 'split should in train, val or test'
        if split == 'train':
            self.data_begin = 0
            self.data = data[0:int(data_len * 0.6)]
        elif split == 'val':
            self.data_begin = int(data_len * 0.6)
            self.data = data[int(data_len * 0.6):int(data_len * 0.8)]
        elif split == 'test':
            self.data_begin = int(data_len * 0.8)
            self.data = data[int(data_len * 0.8):]
        # graph
        self.adj_mat = np.load(os.path.join(data_folder, adj_mat_file))
        # NOTE: symmetrize
        self.adj_mat = np.maximum.reduce([self.adj_mat, self.adj_mat.T]) # (self.adj_mat + self.adj_mat.T) / 2
        self.n_nodes = self.adj_mat.shape[0]
        self.n_edges, self.u_edges, self.u_distance = directed_physical_graph(self.adj_mat, squared_dist=True)
        self.u_edges = torch.Tensor(self.u_edges).type(torch.long)#, dtype=torch.long)
        n_edges = self.u_edges.size(0)
        dic = Counter([(self.u_edges[i,0], self.u_edges[i,1]) for i in range(n_edges)])
        assert max(list(dic.values())), 'distance graph asymmetric'

        self.u_distance = torch.Tensor(self.u_distance)
        self.d_edges = torch.cat([self.u_edges, torch.arange(0, self.n_nodes)[:,None] + torch.zeros((2,), dtype=torch.long)], 0)
        self.graph_info = {
            'n_nodes': self.n_nodes,
            'u_edges': self.u_edges,
            'u_dist': self.u_distance
        }
        # print(self.d_edges)

    def __len__(self):
        return (self.data.shape[0] - self.T) // self.stride

    def __getitem__(self, index):
        y = self.data[index * self.stride:index * self.stride + self.t] # in (t, n_nodes, n_channels)
        x = self.data[index * self.stride:index * self.stride + self.T] # in (T, n_nodes, n_channels)
        # model(y) = x
        if self.use_one_channel:
            x = x[...,0:1]
        time = torch.arange(0, self.T).type(torch.long) + index * self.stride + self.data_begin
        if self.return_time:
            return torch.Tensor(y), torch.Tensor(x), time
        else:
            return torch.Tensor(y), torch.Tensor(x)
        

class WeatherDataset(Dataset):
    def __init__(self, data_folder, adj_file, data_file, T, t=10, stride=1, split='train', return_time=False, use_one_channel=False) -> None:
        '''
        train:val:test = 6:2:2
        Components:
            data: in (T_total, n_nodes, n_channels)
        '''
        super().__init__()
        # time series
        self.T = T
        self.t = t
        assert T - t < 6, 'T - t should be in 1, 2, 3, 4, 5'
        self.stride = stride
        self.return_time = return_time
        self.use_one_channel = use_one_channel

        data = pickle.load(open(os.path.join(data_folder, data_file), 'rb'))
        data = np.expand_dims(data['all'], axis=-1) # ndarray in (T_total, n_nodes, 1)
        
        print('nan_count', len(data[np.isnan(data)]))
        # print('datashape', data.shape, data[0:2])
        self.signal_channel = data.shape[-1]
        data_len = data.shape[0]
        # print('dat_len', data_len)
        assert split in ['train', 'val', 'test'], 'split should in train, val or test'
        if split == 'train':
            self.data_begin = 0
            self.data = data[0:int(data_len * 0.35)]
        elif split == 'val':
            self.data_begin = int(data_len * 0.35)
            self.data = data[int(data_len * 0.35):int(data_len * 0.5)]
        elif split == 'test':
            self.data_begin = int(data_len * 0.5)
            self.data = data[int(data_len * 0.5):]
        
        # graph
        self.adj_mat = np.load(os.path.join(data_folder, adj_file))
        self.n_nodes = self.adj_mat.shape[0]
        self.n_edges, self.u_edges, self.u_distance = directed_physical_graph(self.adj_mat, squared_dist=False)
        self.u_edges = torch.Tensor(self.u_edges).type(torch.long)
        self.u_distance = torch.Tensor(self.u_distance)
        self.d_edges = torch.cat([self.u_edges, torch.arange(0, self.n_nodes)[:,None] + torch.zeros((2,), dtype=torch.long)], 0)
        self.graph_info = {
            'n_nodes': self.n_nodes,
            'u_edges': self.u_edges,
            'u_dist': self.u_distance
        }
        # print(self.d_edges)
    
    def __len__(self):
        return self.data.shape[0] - self.T
    
    def __getitem__(self, index):
        y = self.data[index * self.stride:index * self.stride + self.t] # in (t, n_nodes, n_channels)
        x = self.data[index * self.stride:index * self.stride + self.T] # in (T, n_nodes, n_channels)
        # model(y) = x
        if self.use_one_channel:
            x = x[...,0:1]
        time = torch.arange(0, self.T).type(torch.long) + index + self.data_begin
        if self.return_time:
            return torch.Tensor(y), torch.Tensor(x), time
        else:
            return torch.Tensor(y), torch.Tensor(x)
        



