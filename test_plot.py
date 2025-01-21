# Load existing model and plot the results
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lib.unrolling_model import UnrollingModel
# from lib.graph_learning_module import Swish
from lib.backup_modules import visualise_graph
from tqdm import tqdm
import os
import math
from utils import *
import argparse
from collections import Counter
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', help='CUDA device', type=int)
parser.add_argument('--dataset', help='dataset name', type=str)
parser.add_argument('--batchsize', help='batch size', type=int)
parser.add_argument('--tin', help='time input', default=12, type=int)
parser.add_argument('--tout', help='time output', default=12, type=int)
parser.add_argument('--hop', help='k for kNN', default=6, type=int)
parser.add_argument('--numblock', help='number of admm blocks', default=5, type=int)
parser.add_argument('--numlayer', help='number of admm layers', default=25, type=int)
parser.add_argument('--cgiter', help='CGD iterations', default=3, type=int)
parser.add_argument('--seed', help='random seed', default=3407, type=int)

parser.add_argument('--mode', help='normalization mode', default='normalize', type=str)

parser.add_argument('--ablation', help='operator to elimnate in ablation study', default='None', type=str)#action='store_true', help='run ablation model')

parser.add_argument('--extrapolation', help='use extrapolation', action='store_true')#, default=False)
parser.set_defaults(extrapolation=False)
parser.add_argument('--modelpath', help='model path', default=None, type=str)
args = parser.parse_args()

seed_everything(args.seed)
# Hyper-parameter[s
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
batch_size = args.batchsize
num_workers = 4


k_hop = args.hop
dataset_dir = '../datasets/PEMS0X_data/'
experiment_name = f'{k_hop}_hop_seed{args.seed}'
if args.ablation != 'None':
    experiment_name = f'wo_{args.ablation}' + experiment_name
if not args.extrapolation:
    experiment_name = 'LR_' + experiment_name
dataset_name = args.dataset
T = args.tin + args.tout
t_in = args.tin
stride = 3

return_time = True

train_set, val_set, test_set, train_loader, val_loader, test_loader = create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time)
signal_channels = train_set.signal_channel

print('signal channels', signal_channels)
# data normalization
data_normalization = Normalization(train_set, args.mode, device)

num_admm_blocks = args.numblock
num_heads = 4
feature_channels = 6
ADMM_info = {
                 'ADMM_iters':args.numlayer,
                 'CG_iters': args.cgiter,
                 'PGD_iters': 3,
                 'mu_u_init':3,
                 'mu_d1_init':3,
                 'mu_d2_init':3,
                 }
# graph_sigma = 6

model_pretrained_path = args.modelpath


print('args.ablation', args.ablation)
model = UnrollingModel(num_admm_blocks, device, T, t_in, num_heads, train_set.signal_channel, feature_channels, GNN_layers=2, graph_info=train_set.graph_info, ADMM_info=ADMM_info, k_hop=k_hop, ablation=args.ablation, use_extrapolation=args.extrapolation).to(device)
# 'UnrollingForecasting/MainExperiments/models/v2/PEMS04/direct_4b_4h_6f/val_15.pth'

if model_pretrained_path is not None:
    model = change_model_location(model_pretrained_path, device)
    # for name, param in model.named_parameters():
    #     print(name)
    #     if 'device' in name:
    #         print(name, param.data)
    #         param.data = device # param.data.to('cuda:0')
    # model_args_path = model_pretrained_path.replace('val', 'args')
    # torch.save(model.state_dict(), model_pretrained_path)
    # model.load_state_dict(torch.load(model_args_path))
    # if hasattr(model, 'device'):
    #     model.device = device
    # for submodule in model.children():
    #     if hasattr(submodule, 'device'):
    #         submodule.device = device
    #         for subsubmodule in model.children():
    #             if hasattr(subsubmodule, 'device'):
    #                 subsubmodule.device = device
    print(f'loaded model in {model_pretrained_path}')

ADMM_iters = ADMM_info['ADMM_iters']
# optimizer


# 创建文件处理器
log_dir = f'../JanModified/test_logs_midparam/{experiment_name}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{dataset_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.log'

logger = setup_logger('logger1', os.path.join(log_dir, log_filename), logging.DEBUG, to_console=True)

print('log dir', log_dir)
logger.info('#################################################')
print(" ".join(sys.argv))
logger.info('PARAMETER SETTINGS:')
for arg, value in vars(args).items():
    logger.info("%s: %s", arg, value)
logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# logger.info('model path: %s', model_pretrained_path)
# logger.info(f'learning k hop: {k_hop}')
logger.info('feature channels: %d', feature_channels)
# logger.info(f'graph sigma: {graph_sigma}')
# logger.info(f'batch size: {batch_size}')
# logger.info(f'learning rate: {learning_rate}')
logger.info("Total parameters: %d", total_params)
# logger.info(f'device: {device}')
logger.info('PARAMTER SETTINGS:')
logger.info('ADMM blocks: %d', num_admm_blocks)
logger.info('ADMM info: %s', ADMM_info)
logger.info('graph info: nodes %d, edges %d, signal channels %d', train_set.n_nodes, train_set.n_edges, signal_channels)
logger.info('--------BEGIN TRAINING PROCESS------------')

metrics, metrics_d = test(model, test_loader, data_normalization, False, logger, args, device, signal_channels)

logger.info('Test (ALL): rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', metrics['rec_RMSE'], metrics['pred_RMSE'], metrics['pred_MAE'], metrics['pred_MAPE'])
for i, s in enumerate(['flow', 'occupancy', 'speed']):
    logger.info('Test (%s): rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', s, metrics_d['rec_RMSE'][i], metrics_d['pred_RMSE'][i], metrics_d['pred_MAE'][i], metrics_d['pred_MAPE'][i])