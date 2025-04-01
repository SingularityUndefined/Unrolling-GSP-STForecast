import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TrafficDataset
import os
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time, use_one_channel=False):
    data_folder = os.path.join(dataset_dir, dataset_name)
    graph_csv = dataset_name + '.csv'
    data_file = dataset_name + '.npz'
    if dataset_name == 'PEMS03':
        id_file = dataset_name + '.txt'
    else:
        id_file = None

    train_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'train', id_file=id_file, return_time=return_time, use_one_channel=use_one_channel)
    val_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'val', id_file=id_file, return_time=return_time, use_one_channel=use_one_channel)
    test_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'test', id_file=id_file, return_time=return_time, use_one_channel=use_one_channel)

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


class Normalization():
    def __init__(self, dataset:TrafficDataset, mode:str, device):
        assert mode in ['normalize', 'standardize'], 'mode should be in [normalize, standardize]'
        self.mode = mode
        if mode == 'standardize':
            self.mean = torch.Tensor(dataset.data.mean(0)).to(device)
            self.std = torch.Tensor(dataset.data.std(0)).to(device)
        elif mode == 'normalize':
            self.min = torch.Tensor(dataset.data.min(0)).to(device)
            self.max = torch.Tensor(dataset.data.max(0)).to(device)

    def normalize_data(self, x, use_one_channel=False):
        '''
        *args = (mean, std) or *args = (min, max)
        '''
        if self.mode == 'standardize':
            if use_one_channel:
                return (x - self.mean[...,0:1]) / self.std[...,0:1]
            else:
                return (x - self.mean) / self.std
        elif self.mode == 'normalize':
            if use_one_channel:
                return (x - self.min[...,0:1]) / (self.max[...,0:1] - self.min[...,0:1])
            else:
                return (x - self.min) / (self.max - self.min)
        
    def recover_data(self, x, use_one_channel=False):
        if use_one_channel:
            if self.mode == 'standardize':
                return x * self.std[...,0:1] + self.mean[...,0:1]
            elif self.mode == 'normalize':
                return x * (self.max[...,0:1] - self.min[...,0:1]) + self.min[...,0:1]
        else:
            if self.mode == 'standardize':
                return x * self.std + self.mean
            elif self.mode == 'normalize':
                return x * (self.max - self.min) + self.min
        
def plot_loss_curve(train_loss, val_loss, save_path, val_freq=5, use_log=False):
    train_len, val_len = len(train_loss), len(val_loss)
    if train_len > 1:
        plt.figure()
        plt.plot(list(range(1, train_len + 1)), train_loss, label='train')
    if val_len != 0:
        plt.plot(list(range(val_freq, val_len * val_freq + 1, val_freq)), val_loss, label='val')
    if use_log:
        plt.yscale('log')
    if train_len > 1:
        plt.legend()
        plt.savefig(save_path)
        plt.close()

def log_gradients(epoch, num_epochs, iteration_count, train_loader, model, grad_logger, args):
    if (iteration_count + 1) % args.loggrad == 0:
        grad_logger.info(f'[Epoch {epoch}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]')
    # when debug, print every (3 * args.loggrad) iterations to console
    if (iteration_count + 1) % (3 * args.loggrad) == 0 and args.debug:
        print(f'[Epoch {epoch}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]')

    for name, param in model.named_parameters():
        if 'agg' in name and 'weight' in name:
            if (iteration_count + 1) % args.loggrad == 0:
                grad_logger.info(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')
            if (iteration_count + 1) % (3 * args.loggrad) == 0 and args.debug:
                print(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')

        if model.use_extrapolation and model.use_old_extrapolation and 'linear_extrapolation' in name:
            if (iteration_count + 1) % args.loggrad == 0:
                grad_logger.info(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')
            if (iteration_count + 1) % (3 * args.loggrad) == 0 and args.debug:
                print(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')

def print_gradients(model):
    for name, param in model.named_parameters():
        if 'agg' in name and 'weight' in name:
            print(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')
        if model.use_extrapolation and model.use_old_extrapolation and 'linear_extrapolation' in name:
            print(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')

def change_model_location(model_path, device):
    model_params = torch.load(model_path, map_location=device)
    model.load_state_dict(model_params)
    # model = torch.load(model_path, map_location=device).to(device)
    for name, module in model.named_children():
        if hasattr(module, 'device'):
            print(f'Loaded module: {name}')
            module.device = device
        if name == 'model_blocks':
            for block in module:
                block['feature_extractor'].device = device
                block['ADMM_block'].device = device
                block['graph_learning_module'].device = device
    return model


def test(model, val_loader, data_normalization, masked_flag, args, device, signal_channels, mode='test', loss_fn=None, use_one_channel=False):
    model.eval()
    with torch.no_grad():
        rec_mse = 0
        pred_mse = 0
        pred_mape = 0
        pred_mae = 0
        if not use_one_channel:
            rec_mse_d = np.zeros((signal_channels,))# .to(device)
            pred_mse_d = np.zeros((signal_channels,))# .to(device)
            pred_mape_d = np.zeros((signal_channels,))# .to(device)
            pred_mae_d = np.zeros((signal_channels,))# .to(device)

        if mode == 'val':
            running_loss = 0

        for y, x, t_list in tqdm(val_loader):
            y, x, t_list = y.to(device), x.to(device), t_list.to(device)

            # y = (y - train_mean) / train_std
            # y = (y - train_min) / (train_max - train_min)
            normed_y = data_normalization.normalize_data(y)
            normed_x = data_normalization.normalize_data(x, args.use_one_channel)
            normed_output = model(normed_y, t_list)
            
            if args.normed_loss:
                if loss_fn is not None:
                    if masked_flag:
                        loss = loss_fn(normed_output[:, args.tin:], normed_x[:, args.tin:])
                    else:
                        loss = loss_fn(normed_output, normed_x)
                    running_loss += loss.item()
                # recover data
                output = data_normalization.recover_data(normed_output, args.use_one_channel)

            else:
                output = data_normalization.recover_data(normed_output, args.use_one_channel)
                if loss_fn is not None:
                    if masked_flag:
                        loss = loss_fn(output[:,args.tin:], x[:,args.tin:])
                    else:
                        loss = loss_fn(output, x)
                    running_loss += loss.item()
            
            # if args.mode == 'normalize':
            #     output = nn.ReLU()(output)
            # output = output * (train_max - train_min) + train_min
            # output = output * train_std + train_mean

            rec_mse += ((x[:,:args.tin] - output[:,:args.tin]) ** 2).mean().item()
            if not use_one_channel:
                rec_mse_d += ((x[:,:args.tin] - output[:,:args.tin]) ** 2).mean((0,1,2)).cpu().numpy()# .item()
            if masked_flag:
                x, output = x[:,args.tin:], output[:,args.tin:]
            
            # if loss_fn is not None:
            #     loss = loss_fn(output, x)
            #     running_loss += loss.item()

            
            # x, output = x[:,:,:,1], output[:,:,:,1]
            if masked_flag:
                pred_mse += ((x - output) ** 2).mean().item()
                pred_mae += (torch.abs(output - x)).mean().item()
                mask = (x > 1e-8)
                pred_mape += (torch.abs(output[mask] - x[mask]) / x[mask]).mean().item() * 100
            else:
                x_pred = x[:,args.tin:]
                output_pred = output[:,args.tin:]
                mask = x_pred > 1e-8
                pred_mse += ((x_pred - output_pred) ** 2).mean().item()
                pred_mae += (torch.abs(output_pred - x_pred)).mean().item()
                pred_mape += (torch.abs(output_pred[mask] - x_pred[mask]) / x_pred[mask]).mean().item() * 100

                if not use_one_channel:
                    pred_mse_d += ((x_pred - output_pred) ** 2).mean((0,1,2)).cpu().numpy()
                    pred_mae_d += (torch.abs(output_pred - x_pred)).mean((0,1,2)).cpu().numpy()
                    for i in range(signal_channels):
                        mask_i = x_pred[:,:,:,i] > 1e-8
                        pred_mape_d[i] += (torch.abs(output_pred[:,:,:,i][mask_i] - x_pred[:,:,:,i][mask_i]) / x_pred[:,:,:,i][mask_i]).mean().item() * 100
            # break

    rec_rmse = math.sqrt(rec_mse / len(val_loader))
    pred_rmse = math.sqrt(pred_mse / len(val_loader))
    pred_mae = pred_mae / len(val_loader)
    pred_mape = pred_mape / len(val_loader)

    if not use_one_channel:
        rec_rmse_d = np.sqrt(rec_mse_d / len(val_loader))
        pred_mse_d = np.sqrt(pred_mse_d / len(val_loader))
        pred_mae_d = pred_mae_d / len(val_loader)
        pred_mape_d = pred_mape_d / len(val_loader)

    if mode == 'val':
        running_loss /= len(val_loader)

    metrics = {
        'rec_RMSE': rec_rmse,
        'pred_RMSE': pred_rmse,
        'pred_MAE': pred_mae,
        'pred_MAPE': pred_mape 
    }

    if not use_one_channel:
        metrics_d = {
            'rec_RMSE': rec_rmse_d,
            'pred_RMSE': pred_mse_d,
            'pred_MAE': pred_mae_d,
            'pred_MAPE': pred_mape_d 
        }

    if mode == 'val':
        if not use_one_channel:
            return running_loss, metrics, metrics_d
        else:
            return running_loss, metrics
    elif mode == 'test':
        if not use_one_channel:
            return metrics, metrics_d
        else:
            return metrics
    # return running_loss