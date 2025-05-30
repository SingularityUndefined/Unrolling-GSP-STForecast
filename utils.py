import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TrafficDataset, DirectedTrafficDataset, WeatherDataset
import os
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import argparse


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time, use_one_channel=False, truncated=False):
    data_folder = os.path.join(dataset_dir, dataset_name)
    graph_csv = dataset_name + '.csv'
    data_file = dataset_name + '.npz'
    if dataset_name == 'PEMS03':
        id_file = dataset_name + '.txt'
    else:
        id_file = None

    train_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'train', id_file=id_file, return_time=return_time, use_one_channel=use_one_channel, truncated=truncated)
    val_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'val', id_file=id_file, return_time=return_time, use_one_channel=use_one_channel, truncated=truncated)
    test_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'test', id_file=id_file, return_time=return_time, use_one_channel=use_one_channel, truncated=truncated)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader

def create_directed_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time, use_one_channel=False):
    data_folder = os.path.join(dataset_dir, dataset_name)
    
    assert dataset_name in ['PEMS-BAY', 'METR-LA']
    if dataset_name == "PEMS-BAY":
        adj_mat_name = 'pems_adj_mat.npy'
        data_file_name = 'pems_node_values.npy'
    else: # dataset_name == "METR-LA"
        adj_mat_name = 'adj_mat.npy'
        data_file_name = 'node_values.npy'
    
    train_set = DirectedTrafficDataset(data_folder, adj_mat_name, data_file_name, T, t_in, stride, 'train', return_time=return_time, use_one_channel=use_one_channel)
    val_set = DirectedTrafficDataset(data_folder, adj_mat_name, data_file_name, T, t_in, stride, 'val', return_time=return_time, use_one_channel=use_one_channel)
    test_set = DirectedTrafficDataset(data_folder, adj_mat_name, data_file_name, T, t_in, stride, 'test', return_time=return_time, use_one_channel=use_one_channel)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader

def create_weather_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time, use_one_channel=False):
    data_folder = os.path.join(dataset_dir, dataset_name, 'dataset/processed')
    assert dataset_name in ['Molene', 'NOAA']
    if dataset_name == 'Molene':
        # data file names
        data_filename = 'dataset_w=10_steps=[1, 2, 3, 4, 5]_splits=[0.35, 0.15, 0.5].pickle'
        adj_filename = 'weighted_adjacency.npy'
    else: # NOAA
        # datafile names
        data_filename = 'NOA_w=10_steps=[1, 2, 3, 4, 5]_splits=[0.35, 0.15, 0.5].pickle'
        adj_filename = 'weighted_adj.npy'
    
    # load the dataset
    train_set = WeatherDataset(data_folder, adj_filename, data_filename, T, t_in, stride, 'train', return_time=return_time, use_one_channel=use_one_channel)
    val_set = WeatherDataset(data_folder, adj_filename, data_filename, T, t_in, stride, 'val', return_time=return_time, use_one_channel=use_one_channel)
    test_set = WeatherDataset(data_folder, adj_filename, data_filename, T, t_in, stride, 'test', return_time=return_time, use_one_channel=use_one_channel)
    # create dataloaders
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

def change_model_location(model, model_path, device):
    model_params = torch.load(model_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(model_params, strict=False)
    print('missing keys:', missing_keys)
    print('unexpected keys:', unexpected_keys)
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


def test(model, val_loader, data_normalization, masked_flag, config, device, signal_channels, mode='test', loss_fn=None, use_one_channel=False, use_tqdm=True):
    model.eval()
    batch_count = 0
    all_zero_batchs = 0
    t_out = config['model']['t_out']
    t_in = config['model']['t_in']
    output_list = []
    x_list = []
    with torch.no_grad():
        rec_mse = 0
        pred_mse = 0
        pred_mape = 0
        pred_mae = 0
        nearest_loss = 0
        pred_mse_stepwise = torch.zeros((t_out,))
        truth_sq_stepwise = torch.zeros((t_out,))
        truth_sq = 0
        if not use_one_channel:
            rec_mse_d = np.zeros((signal_channels,))# .to(device)
            pred_mse_d = np.zeros((signal_channels,))# .to(device)
            pred_mape_d = np.zeros((signal_channels,))# .to(device)
            pred_mae_d = np.zeros((signal_channels,))# .to(device)

        if mode == 'val':
            running_loss = 0

        if use_tqdm:
            val_loader_iter = tqdm(val_loader)
        else:
            val_loader_iter = val_loader

        for y, x, t_list in val_loader_iter:
            # if batch_count < 120:
            #     batch_count += 1
            #     continue
            y, x, t_list = y.to(device), x.to(device), t_list.to(device)

            # y = (y - train_mean) / train_std
            # y = (y - train_min) / (train_max - train_min)
            if data_normalization is not None:
                normed_y = data_normalization.normalize_data(y)
                normed_x = data_normalization.normalize_data(x, config['model']['use_one_channel'])
                normed_output = model(normed_y, t_list)
                
                if config['normed_loss']:
                    if loss_fn is not None:
                        if masked_flag:
                            loss = loss_fn(normed_output[:, config['model']['t_in']:], normed_x[:, config['model']['t_in']:])
                        else:
                            loss = loss_fn(normed_output, normed_x)
                        running_loss += loss.item()
                    # recover data
                    output = data_normalization.recover_data(normed_output, config['model']['use_one_channel'])

                else:
                    output = data_normalization.recover_data(normed_output, config['model']['use_one_channel'])
                    if loss_fn is not None:
                        if masked_flag:
                            loss = loss_fn(output[:,config['model']['t_in']:], x[:,config['model']['t_in']:])
                        else:
                            loss = loss_fn(output, x)
                        running_loss += loss.item()
            else:
                output = model(y, t_list)
                if loss_fn is not None:
                    if masked_flag:
                        loss = loss_fn(output[:,config['model']['t_in']:], x[:,config['model']['t_in']:])
                    else:
                        loss = loss_fn(output, x)
                    running_loss += loss.item()
            
            # if args.mode == 'normalize':
            #     output = nn.ReLU()(output)
            # output = output * (train_max - train_min) + train_min
            # output = output * train_std + train_mean
            # if data_normalization is not None:
            # metrics
            '''
            metrics_batch = compute_metrics(output.detach().cpu(), x.detach().cpu(), masked_flag, t_in)
            rec_mse += metrics_batch['rec_MSE'] # ((x[:,:config['model']['t_in']] - output[:,:config['model']['t_in']]) ** 2).detach().cpu().mean().item()
            pred_mse += metrics_batch['pred_MSE']
            pred_mae += metrics_batch['pred_MAE']
            if metrics_batch['pred_MAPE'] is None:
                print('exist all zero batchs in ground-truth in pred_mape')
                all_zero_batchs += 1
            else:
                pred_mape += metrics_batch['pred_MAPE']
            pred_mse_stepwise += metrics_batch['pred_MSE_stepwise']
            truth_sq_stepwise += metrics_batch['truth_sq_stepwise']
            truth_sq += metrics_batch['truth_sq']
            nearest_loss += metrics_batch['nearest_loss']
            '''
            output_list.append(output.detach().cpu())
            x_list.append(x.detach().cpu())
            if not use_one_channel:
                rec_mse_d += ((x[:,:config['model']['t_in']] - output[:,:config['model']['t_in']]) ** 2).detach().cpu().mean((0,1,2)).cpu().numpy()# .item()
            if masked_flag:
                x, output = x[:,config['model']['t_in']:], output[:,config['model']['t_in']:]
            
            # if loss_fn is not None:
            #     loss = loss_fn(output, x)
            #     running_loss += loss.item()

            
            # x, output = x[:,:,:,1], output[:,:,:,1]
            
            if not masked_flag:
                x_pred = x[:,t_in:]
                output_pred = output[:,t_in:]
                
                # if mask.sum() == 0:
                #     print(x_pred, output_pred)

                # print('pred_mape', pred_mape)

                if not use_one_channel:
                    pred_mse_d += ((x_pred - output_pred) ** 2).detach().mean((0,1,2)).cpu().numpy()
                    pred_mae_d += (torch.abs(output_pred - x_pred)).detach().mean((0,1,2)).cpu().numpy()
                    for i in range(signal_channels):
                        mask_i = x_pred[:,:,:,i] > 1e-8
                        pred_mape_d[i] += (torch.abs(output_pred[:,:,:,i][mask_i] - x_pred[:,:,:,i][mask_i]) / x_pred[:,:,:,i][mask_i]).detach().cpu().mean().item() * 100
            # break
    '''
    rec_rmse = math.sqrt(rec_mse / len(val_loader))
    pred_rmse = math.sqrt(pred_mse / len(val_loader))
    pred_mae = pred_mae / len(val_loader)
    pred_mape = pred_mape / (len(val_loader) - all_zero_batchs) #len(val_loader)
    pred_rnmse_stepwise = torch.sqrt(pred_mse_stepwise / truth_sq_stepwise)
    pred_rnmse = torch.sqrt(pred_mse / truth_sq)
    '''
    full_output = torch.cat(output_list, 0)
    full_x = torch.cat(x_list, 0)
    metrics = compute_metrics(full_output, full_x, masked_flag, t_in)
    metrics['rNMSE_stepwise'] = torch.sqrt(metrics['pred_MSE_stepwise'] / metrics['truth_sq_stepwise'])
    metrics['rNMSE'] = math.sqrt(metrics['pred_MSE'] / metrics['truth_sq'])
    metrics['rec_RMSE'] = math.sqrt(metrics['rec_MSE'])
    metrics['pred_RMSE'] = math.sqrt(metrics['pred_MSE'])


    if not use_one_channel:
        rec_rmse_d = np.sqrt(rec_mse_d / len(val_loader))
        pred_mse_d = np.sqrt(pred_mse_d / len(val_loader))
        pred_mae_d = pred_mae_d / len(val_loader)
        pred_mape_d = pred_mape_d / (len(val_loader) - all_zero_batchs) #len(val_loader)

    if mode == 'val':
        running_loss /= len(val_loader)
    '''
    metrics = {
        'rec_RMSE': rec_rmse,
        'pred_RMSE': pred_rmse,
        'pred_MAE': pred_mae,
        'pred_MAPE': pred_mape,
        'rNMSE_stepwise': pred_rnmse_stepwise,
        'rNMSE': pred_rnmse,
    }
    '''

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

def compute_metrics(output, x, masked_flag, t_in):
    """
    Compute the metrics for the model
    output: model output, in (B, T, N, C)
    x: ground truth, in (B, T, N, C)
    masked_flag: whether to use the masked loss
    """
    rec_mse_batch = ((x[:,:t_in] - output[:,:t_in]) ** 2).mean().item()

    if masked_flag:
        x, output = x[:,t_in:], output[:,t_in:]
        pred_mse = ((x - output) ** 2).mean().item()
        pred_mae = (torch.abs(output - x)).mean().item()
        mask = (torch.abs(x) > 1e-8)
        if mask.sum() > 0:
            pred_mape = (torch.abs(output[mask] - x[mask]) / torch.abs(x[mask])).mean().item() * 100
        else:
            pred_mape = None
            # print('exist all zero batchs in ground-truth in pred_mape')
            # print(x)
        
        # rnmse metric
        pred_mse_stepwise = ((x - output) ** 2).mean((0,2,3))
        truth_sq_stepwise = (x ** 2).mean((0,2,3))
        truth_sq = (x ** 2).mean()
        nearest_loss = ((x[:, 0] - output[:, 0]) ** 2).mean().item()
        # rnmse_stepwise = torch.sqrt(mse_stepwise / truth_stepwise)
        # rnmse = torch.sqrt(mse_stepwise.mean() / truth_stepwise.mean())


    else:
        x_pred = x[:,t_in:]
        output_pred = output[:,t_in:]
        mask = (torch.abs(x_pred) > 1e-8)
        pred_mse = ((x_pred - output_pred) ** 2).mean().item()
        pred_mae = (torch.abs(output_pred - x_pred)).mean().item()
        if mask.sum() > 0:
            pred_mape = (torch.abs(output_pred[mask] - x_pred[mask]) / torch.abs(x_pred[mask])).mean().item() * 100
        else:
            pred_mape = None
            # print('exist all zero batchs in ground-truth in pred_mape')
            # print(x_pred)
        # rnmse metric
        pred_mse_stepwise = ((x_pred - output_pred) ** 2).mean((0,2,3))
        truth_sq_stepwise = (x_pred ** 2).mean((0,2,3))
        truth_sq = (x_pred ** 2).mean()
        nearest_loss = ((x[:, t_in] - output[:, t_in]) ** 2).mean().item()
        # rnmse_stepwise = torch.sqrt(mse_stepwise / truth_stepwise)
        # rnmse = torch.sqrt(mse_stepwise.mean() / truth_stepwise.mean())
    
    # dictionary for metrics
    metrics = {
        'rec_MSE': rec_mse_batch,
        'pred_MSE': pred_mse,
        'pred_MAE': pred_mae,
        'pred_MAPE': pred_mape,
        'pred_MSE_stepwise': pred_mse_stepwise,
        'truth_sq_stepwise': truth_sq_stepwise,
        'truth_sq': truth_sq,
        'nearest_loss': nearest_loss
    }

    return metrics



# check the gradients
def check_nan_gradients(model:nn.Module):
    # print all gradients
    # flag = False
    # nan_module = None # list of parameters with NaN gradients and inf gradients
    for name, param in reversed(list(model.named_parameters())):
        if param.grad is not None:
            param_detached = param.detach().cpu()
            grad_detached = param.grad.detach().cpu()
            if torch.isnan(grad_detached).any() or torch.isnan(param_detached).any() or torch.isinf(grad_detached).any() or torch.isinf(param_detached).any():
                return name
                # break
    return

def log_parameters_scalars(model:nn.Module, name_list:list):
    # num_blocks = model.num_blocks
    param_dicts = {}
    grad_dict = {}
    for check_name in name_list:
        param_dicts[check_name] = {}
        # grad_dicts[check_name] = {}
        for name, param in model.named_parameters():
            if check_name in name:
                name_split = name.split('.')
                block_id, param_name = name_split[1], name_split[-1]
                param_dicts[check_name][f'{param_name}_{block_id}_min'] = param.detach().cpu().min().item()
                param_dicts[check_name][f'{param_name}_{block_id}_max'] = param.detach().cpu().max().item()
                grad_dict[f'{param_name}_{block_id}'] = param.grad.data.detach().cpu().norm(2).item()
                # logger.info(f'\t {name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')
    return param_dicts, grad_dict

def dataframe_from_tensorboard(log_dir, selected_tag):
    """
    处理 TensorBoard 日志数据
    :param log_dir: TensorBoard 日志目录
    :return: 返回处理后的数据
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 获取所有的 tags
    tags = ea.Tags()
    print("Available tags:", tags)

    # 提取某个 tag 的数据（例如 'train/loss'）
    scalar_data = ea.Scalars(selected_tag)

    # 将数据转换为 DataFrame

    df = pd.DataFrame({
        "step": [e.step for e in scalar_data],
        "value": [e.value for e in scalar_data]
    })

    return df

def plot_dataframe(df, title="Training Loss Curve"):
    """
    绘制 DataFrame 数据
    :param df: DataFrame 数据
    :param title: 图表标题
    """

    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["value"])
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.show()


def generate_experiment_name(args: argparse.Namespace, config:dict):
    pass

def log_tensorboard():
    pass

