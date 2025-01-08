import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TrafficDataset
import os
import logging

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time):
    data_folder = os.path.join(dataset_dir, dataset_name)
    graph_csv = dataset_name + '.csv'
    data_file = dataset_name + '.npz'
    if dataset_name == 'PEMS03':
        id_file = dataset_name + '.txt'
    else:
        id_file = None

    train_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'train', id_file=id_file, return_time=return_time)
    val_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'val', id_file=id_file, return_time=return_time)
    test_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'test', id_file=id_file, return_time=return_time)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader

import logging

def setup_logger(name, logfile, level=logging.INFO, to_console=False):
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建文件handler并设置日志级别和格式
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 根据参数决定是否创建控制台handler
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# def 
class WeightedMSELoss(nn.Module):
    def __init__(self, t, T, weights=None) -> None:
        super().__init__()
        self.t = t
        self.T = T
        if weights is None:
            self.weights = t / (T - t)
        else:
            self.weights = weights

    def forward(self, inputs, target):
        rec_loss = nn.MSELoss()(inputs[:,:self.t], target[:,:self.t])# ((inputs[:,:self.t] - target[:,:self.t]) ** 2).mean()
        pred_loss = nn.MSELoss()(inputs[:,self.t:], target[:,self.t:])# ((inputs[:,self.t:] - target[:,self.t:]) ** 2).mean()
        return rec_loss * self.weights + pred_loss