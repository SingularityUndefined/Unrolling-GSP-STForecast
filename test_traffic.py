# import

import torch
from torch.utils.data import DataLoader, Dataset
from lib.unrolling_model import UnrollingModel
from tqdm import tqdm
import os
import yaml
import gc
import argparse
from utils import *

import math

# argparse
with open("config.yaml", 'r') as f: 
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', help='CUDA device', default=-1, type=int)
parser.add_argument('--dataset', help='dataset name', type=str)
parser.add_argument('--batchsize', help='batch size', type=int)
parser.add_argument('--mode', help='normalization mode', default='standardize', type=str)
parser.add_argument('--path', type=str)

parser.add_argument('--neighbors', help='kNN neighbors', default=config['model']['kNN'], type=int)
parser.add_argument('--interval', help='intervals for time graph', default=config['model']['interval'], type=int)
parser.add_argument('--FElayers', help='feature extractor layers', default=1, type=int)

# experiment type
parser.add_argument('--ablation', help='operator to elimnate in ablation study', default='None', type=str)#action='store_true', help='run ablation model')

# model hyper-params
parser.add_argument('--seed', help='random seed', default=3407, type=int)

# optimizer and learning setting

parser.add_argument('--sharedM', dest='sharedM', action='store_true')
parser.set_defaults(sharedM=config['model']['sharedM'])

parser.add_argument('--sharedQ', dest='sharedQ', action='store_true')
parser.set_defaults(sharedQ=config['model']['sharedQ'])

parser.add_argument('--sharedV', dest='diff_interval', action='store_false')
parser.set_defaults(diff_interval=config['model']['diff_interval'])
# training settings


parser.add_argument('--tout', help='t_out', default=config['model']['t_out'], type=int)
parser.add_argument('--trunc', dest='trunc', action='store_true')
parser.set_defaults(trunc=False)
parser.add_argument('--blocks', help='number of blocks in the model', default=config['model']['num_blocks'], type=int)
parser.add_argument('--layers', help='number of layers in the model', default=config['model']['num_layers'], type=int)
parser.add_argument('--CGiters', help='number of CG layers in the model', default=config['model']['CG_iters'], type=int)

parser.add_argument('--predonly', dest='pred_only', action='store_true')
parser.set_defaults(pred_only=False)

args = parser.parse_args()

config['model']['kNN'] = args.neighbors
config['model']['interval'] = args.interval
config['model']['sharedM'] = args.sharedM
config['model']['sharedQ'] = args.sharedQ
config['model']['diff_interval'] = args.diff_interval
config['model']['num_blocks'] = args.blocks
# config['model']['num_layers'] = args.layers
config['model']['num_layers'] = args.layers
config['model']['CG_iters'] = args.CGiters
# config['model']['t_out'] = args.tout

# seed, device, training settings
seed_everything(args.seed)
# Hyper-parameter[
if args.cuda != -1:
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
    
batch_size = args.batchsize
learning_rate = config['learning_rate']
# num_epochs = args.epochs
num_workers = config['num_workers']

# loss
loss_name = config['loss_function']

if loss_name == 'MSE':
    loss_fn = nn.MSELoss()
elif loss_name == 'Huber':
    loss_fn = nn.HuberLoss(delta=1)
elif loss_name == 'Mix':
    loss_fn = WeightedMSELoss(args.tin, args.tin + args.tout)

k_hop = config['model']['kNN']
interval = config['model']['interval']
dataset_dir = '/home/disk/qij/TS_datasets/'
if not os.path.exists(dataset_dir):
    dataset_dir = '../datasets/'

if 'PEMS0' in args.dataset:
    dataset_dir = os.path.join(dataset_dir, 'PEMS0X_data')

experiment_dir = f'lr_{learning_rate:.0e}_seed_{args.seed}'
# experiment_name = f'{k_hop}_hop_{interval}_int_lr_{learning_rate:.0e}_seed{args.seed}'
dataset_name = args.dataset
num_admm_blocks = config['model']['num_blocks']
num_heads = config['model']['num_heads']
interval = config['model']['interval']
feature_channels = config['model']['feature_channels']
ADMM_iters = config['model']['num_layers']

experiment_name = f"{dataset_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f"

if args.pred_only:
    experiment_name = 'predOnly_' + experiment_name
    # config['model']['use_extrapolation'] = False
if args.trunc:
    experiment_name = 'trunc_' + experiment_name
    # config['model']['use_extrapolation'] = False

if args.tout != config['model']['t_out']:
    experiment_name = f'{args.tout}out_' + experiment_name
    config['model']['t_out'] = args.tout

if args.ablation != 'None':
    experiment_name = f'wo_{args.ablation}_' + experiment_name
if not config['model']['use_extrapolation']:
    experiment_name = 'LR_' + experiment_name

if not config['model']['use_one_channel']:
    experiment_name = 'AllChannel_' + experiment_name

if config['model']['sharedM']:
    experiment_name = 'shareM_' + experiment_name

if config['model']['sharedQ']:
    experiment_name = 'shareQ_' + experiment_name

if config['model']['diff_interval']:
    experiment_name = 'diffV_' + experiment_name

if config['normed_loss']:
    experiment_name = experiment_name + '_normed_loss'
else:
    experiment_name = experiment_name + '_true_loss'

experiment_name = os.path.join(experiment_dir, experiment_name)

log_filename = f"nn_{k_hop}_int_{interval}_{loss_name}.log"

T = config['model']['t_in'] + config['model']['t_out']
t_in = config['model']['t_in']
stride = config['data_stride']

return_time = True

# load data
if 'PEMS0' in args.dataset:
    train_set, val_set, test_set, train_loader, val_loader, test_loader = create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time, use_one_channel=config['model']['use_one_channel'], truncated=args.trunc) # use one channel

else:
    train_set, val_set, test_set, train_loader, val_loader, test_loader = create_directed_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time, use_one_channel=config['model']['use_one_channel'])
signal_channels = train_set.signal_channel

# if args.use_one_channel:
#     signal_channels = 1

signal_list = ['flow', 'occupancy', 'speed']

print('number of channels:', signal_channels)
# data normalization
data_normalization = Normalization(train_set, args.mode, device)
# print(train_set.data[...,0].mean(0).min(), train_set.data[...,0].mean(0).max())
if args.mode == 'standardize':
    print('mean value of data', data_normalization.mean[...,0].min(), data_normalization.mean[...,0].max())
    print('std of data', data_normalization.std[...,0].min(), data_normalization.std[...,0].max())
else:
    print('min value of data', data_normalization.min[...,0].min(), data_normalization.min[...,0].max())
    print('max value of data', data_normalization.max[...,0].min(), data_normalization.max[...,0].max())


ADMM_info = {
                 'ADMM_iters':config['model']['num_layers'],
                 'CG_iters': config['model']['CG_iters'],
                 'PGD_iters': config['model']['PGD_iters'],
                 'mu_u_init':config['ADMM_params']['mu_u'],
                 'mu_d1_init':config['ADMM_params']['mu_d1'],
                 'mu_d2_init':config['ADMM_params']['mu_d2'],
                 }
# graph_sigma = 6

model_pretrained_path = args.path


print('args.ablation', args.ablation)
model = UnrollingModel(num_admm_blocks, device, T, t_in, num_heads, interval, train_set.signal_channel, feature_channels, GNN_layers=2, graph_info=train_set.graph_info, ADMM_info=ADMM_info, k_hop=k_hop, ablation=args.ablation, st_emb_info=config['st_emb_info'], use_extrapolation=config['model']['use_extrapolation'], extrapolation_agg_layers=args.FElayers, use_one_channel=config['model']['use_one_channel'], sharedM=config['model']['sharedM'], sharedQ=config['model']['sharedQ'], diff_interval=config['model']['diff_interval'], predict_only=args.pred_only).to(device)

print(model.linear_extrapolation.input_layer.temporal_hist.linear.weight.size())
# 'UnrollingForecasting/MainExperiments/models/v2/PEMS04/direct_4b_4h_6f/val_15.pth'


# model = model.load_state_dict(torch.load(model_pretrained_path))
    # TODO: map to models
print('device', device)
model = change_model_location(model, model_pretrained_path, device)

ADMM_iters = ADMM_info['ADMM_iters']
# test
if not config['model']['use_one_channel']:
    test_loss, test_metrics, test_metric_d = test(model, test_loader, data_normalization, False, config, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=False)
else:
    test_loss, test_metrics = test(model, test_loader, data_normalization, False, config, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=True)

print(model_pretrained_path)
rec_rmse = test_metrics['rec_RMSE']
pred_rmse = test_metrics['pred_RMSE']
pred_mae = test_metrics['pred_MAE']
pred_mape = test_metrics['pred_MAPE']
print(f'Test, Loss:{test_loss:.4f}, rec_RMSE:{rec_rmse:.4f}, RMSE:{pred_rmse:.4f}, MAE:{pred_mae:.4f}, MAPE(%%):{pred_mape:.4f}')

if not config['model']['use_one_channel']:
    test_loss, test_metrics, test_metric_d = test(model, val_loader, data_normalization, False, config, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=False)
else:
    test_loss, test_metrics = test(model, val_loader, data_normalization, False, config, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=True)

print(model_pretrained_path)
rec_rmse = test_metrics['rec_RMSE']
pred_rmse = test_metrics['pred_RMSE']
pred_mae = test_metrics['pred_MAE']
pred_mape = test_metrics['pred_MAPE']
print(f'Val, Loss:{test_loss:.4f}, rec_RMSE:{rec_rmse:.4f}, RMSE:{pred_rmse:.4f}, MAE:{pred_mae:.4f}, MAPE(%%):{pred_mape:.4f}')
