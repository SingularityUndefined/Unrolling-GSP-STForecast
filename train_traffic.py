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
from tensorboardX import SummaryWriter
import yaml
import gc
import copy

with open("config.yaml", 'r') as f: 
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser()
# device and data
parser.add_argument('--cuda', help='CUDA device', default=-1, type=int)
parser.add_argument('--dataset', help='dataset name', type=str)
parser.add_argument('--batchsize', help='batch size', type=int)
parser.add_argument('--mode', help='normalization mode', default='standardize', type=str)

# parameters test
parser.add_argument('--neighbors', help='kNN neighbors', default=config['model']['kNN'], type=int)
parser.add_argument('--interval', help='intervals for time graph', default=config['model']['interval'], type=int)
parser.add_argument('--FElayers', help='feature extractor layers', default=1, type=int)

# experiment type
parser.add_argument('--ablation', help='operator to elimnate in ablation study', default='None', type=str)#action='store_true', help='run ablation model')

# model hyper-params
parser.add_argument('--seed', help='random seed', default=3407, type=int)
parser.add_argument('--debug', dest='debug', help='if debug, save model every iteration', action='store_true')
parser.set_defaults(debug=False)

# optimizer and learning setting
parser.add_argument('--stepLR', dest='use_stepLR', action='store_true')
parser.set_defaults(use_stepLR=False)
parser.add_argument('--stepsize', help='stepLR stepsize', default=8, type=int)
parser.add_argument('--gamma', help='stepLR gamma', default=0.2, type=float)

parser.add_argument('--sharedM', dest='sharedM', action='store_true')
parser.set_defaults(sharedM=config['model']['sharedM'])

parser.add_argument('--sharedQ', dest='sharedQ', action='store_true')
parser.set_defaults(sharedQ=config['model']['sharedQ'])

parser.add_argument('--sharedV', dest='diff_interval', action='store_false')
parser.set_defaults(diff_interval=config['model']['diff_interval'])
# training settings
parser.add_argument('--epochs', help='running epochs', default=70, type=int)
parser.add_argument('--start_epochs', help='start epochs', default=0, type=int)

# log settings
parser.add_argument('--loggrad', help='log gradient norms', default=-1, type=int) # -1 stand for no log, 0 for log all, >0 for log every n iterations

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
num_epochs = args.epochs
num_workers = config['num_workers']

# loss
loss_name = config['loss_function']

if loss_name == 'MSE':
    loss_fn = nn.MSELoss()
elif loss_name == 'Huber':
    loss_fn = nn.HuberLoss(delta=1)
elif loss_name == 'Mix':
    loss_fn = WeightedMSELoss(args.tin, args.tin + args.tout)

def get_degrees(n_nodes, u_edges:torch.Tensor):
    '''
    u_edges: ndarray, in (n_edges, 2), already bidirectional
    '''
    n_edges = u_edges.size(0)
    degrees = np.zeros((n_nodes,), dtype=int)
    for i in range(n_edges):
        degrees[u_edges[i,0]] += 1
    
    counts = Counter(degrees)
    # plot histogram of degree counts, percentage
    plt.figure()
    plt.bar(counts.keys(), counts.values())
    # plt.bar(counts.keys(), counts.values())
    plt.xlabel('Node degree')
    plt.ylabel('Counts')
    plt.title('Degree distribution')
    plt.savefig('degree_distribution.png', dpi=800)

    return counts

# hops
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

model_pretrained_path = None# 'dense_logs_new/models/lr_5e-04_seed_3407/diffV_shareQ_PEMS03_5b25_4h_6f_true_loss/nn_4_int_6_Huber/val_10.pth'


print('args.ablation', args.ablation)
model = UnrollingModel(num_admm_blocks, device, T, t_in, num_heads, interval, train_set.signal_channel, feature_channels, GNN_layers=2, graph_info=train_set.graph_info, ADMM_info=ADMM_info, k_hop=k_hop, ablation=args.ablation, st_emb_info=config['st_emb_info'], use_extrapolation=config['model']['use_extrapolation'], extrapolation_agg_layers=args.FElayers, use_one_channel=config['model']['use_one_channel'], sharedM=config['model']['sharedM'], sharedQ=config['model']['sharedQ'], diff_interval=config['model']['diff_interval'], predict_only=args.pred_only).to(device)
# 'UnrollingForecasting/MainExperiments/models/v2/PEMS04/direct_4b_4h_6f/val_15.pth'
    # TODO: map to models

# best_model = copy.deepcopy(model)

ADMM_iters = ADMM_info['ADMM_iters']

# optimizer
import torch.optim as optim
from torch.optim import lr_scheduler

assert config['optim'] in ['adam', 'adamw'], 'config[\'optim\'] should be adam or adamw'
if config['optim'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif config['optim'] == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])

if args.use_stepLR:
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', args.gamma, 5, cooldown=5, min_lr=5e-6) # StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma) # TODO: step size

# tensorboard logger
tensorboard_logdir = f'./dense_logs_new/TB_log/{experiment_name}/nn_{k_hop}_int_{interval}_{loss_name}'
os.makedirs(tensorboard_logdir, exist_ok=True)
writer = SummaryWriter(tensorboard_logdir)

# create loggers
log_dir = f'./dense_logs_new/train_Logs/{experiment_name}'
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger('logger1', os.path.join(log_dir, log_filename), logging.DEBUG, to_console=True)

if args.loggrad != -1:
    grad_logger_dir = f'./dense_logs_new/grad_logs/{experiment_name}'
    os.makedirs(grad_logger_dir, exist_ok=True)
    grad_logger = setup_logger('logger2', os.path.join(grad_logger_dir, log_filename), logging.INFO, to_console=False)

# model save dir
debug_model_path = os.path.join(f'./dense_logs_new/debug_models/{experiment_name}', f'nn_{k_hop}_int_{interval}_{loss_name}.pth')
model_dir = os.path.join(f'./dense_logs_new/models/{experiment_name}', f'nn_{k_hop}_int_{interval}_{loss_name}')
os.makedirs(model_dir, exist_ok=True)

# print('log dir', log_dir)
logger.info('#################################################')
logger.info("Training CMD:\t"+ " ".join(sys.argv))
logger.info('PARAMETER SETTINGS:')
for arg, value in vars(args).items():
    logger.info("\t %s: %s", arg, value)

logger.info('CONFIG SETTINGS:')
for key, value in config.items():
    if isinstance(value, dict):
        logger.info("\t%s:", key)
        for k, v in value.items():
            logger.info("\t\t%s: %s", k, v)
    else:
        logger.info("\t%s: %s", key, value)
logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('pretrained path: %s', model_pretrained_path)
logger.info('feature channels: %d', feature_channels)
logger.info("Total parameters: %d", total_params)
logger.info('PARAMTER SETTINGS:')
logger.info('ADMM blocks: %d', num_admm_blocks)
logger.info('ADMM info: %s', ADMM_info)
logger.info('graph info: nodes %d, edges %d, signal channels %d', train_set.n_nodes, train_set.n_edges, signal_channels)
logger.info('--------BEGIN TRAINING PROCESS------------')
if args.loggrad != -1:
    grad_logger.info('------BEGIN TRAINING PROCESS-------')
print('log path', os.path.join(log_dir, log_filename))
print('tensorboard log path', tensorboard_logdir)
masked_flag = False
best_val_loss = 20
best_epoch = args.start_epochs

if args.start_epochs > 0:
    model_pretrained_path = os.path.join(model_dir, f'val_{args.start_epochs}.pth')
    model = change_model_location(model, model_pretrained_path, device)
# train models
# test = True
plot_list = f'./dense_logs_new/loss_curves/{experiment_name}'
os.makedirs(plot_list, exist_ok=True)
plot_filename = f'nn_{k_hop}_int_{interval}_{loss_name}.png'
plot_path = os.path.join(plot_list, plot_filename)

train_loss_list = []
val_loss_list = []

for epoch in range(args.start_epochs, num_epochs):
    # TODO: remember to / 50 # don't need now
    model.train()
    running_loss = 0
    rmse_per_time = torch.zeros((T,))# .to(device)
    rmse_sep = 0
    rec_mse = 0
    pred_mse = 0
    pred_mae = 0
    pred_mape = 0
    nearest_loss = 0

    # iteration_count = 0

    for iter_idx, (y, x, t_list) in enumerate(tqdm(train_loader)):
        if iter_idx > 0 and iter_idx % 128 == 0:  # Every 32 iterations
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()
        # print(y.size(), x.size(), t_list.size())
        iteration_count = iter_idx + 1
        optimizer.zero_grad()
        y, x, t_list = y.to(device), x.to(device), t_list.to(device)  # y in (B, t, nodes, full_signal_channels), x in (B, T, N, pred_signal_channels), T in (B, T)
        # normalization

        normed_y = data_normalization.normalize_data(y)
        normed_x = data_normalization.normalize_data(x, config['model']['use_one_channel'])

        try:
            # print('train')
            normed_output = model(normed_y, t_list)  # in (B, T, nodes, 1)
            # print('trained')
            if args.mode == 'normalize':
                normed_output = nn.ReLU()(normed_output) # data_normalization.normalize_data(normed_output, args.use_one_channel)
            # raise ValueError('raised value error')
        except ValueError as ve:
            # plot the loss curve from loss_list
            # plot_loss_curve(loss_list, log_dir)
            logger.error(f'Error in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] - {ve}')
            if args.loggrad != -1:
                grad_logger.error(f'Error in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] - {ve}')

            plot_loss_curve(train_loss_list, val_loss_list, plot_path)

            if not config['model']['use_one_channel']:
                metrics, metrics_d = test(model, test_loader, data_normalization, False, config, device, signal_channels, use_one_channel=False)
            else:
                metrics = test(model, test_loader, data_normalization, False, config, device, signal_channels, use_one_channel=True)

            logger.info('Test (ALL): rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', metrics['rec_RMSE'], metrics['pred_RMSE'], metrics['pred_MAE'], metrics['pred_MAPE'])

            if not config['model']['use_one_channel']:
                for i in range(signal_channels):
                    logger.info('Test (%s): rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', signal_list[i], metrics_d['rec_RMSE'][i], metrics_d['pred_RMSE'][i], metrics_d['pred_MAE'][i], metrics_d['pred_MAPE'][i])

            raise ValueError(f'Error in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] - {ve}') from ve

        # normalized loss
        if config['normed_loss']:
            if masked_flag:
                loss = loss_fn(normed_output[:, t_in:], normed_x[:, t_in:])
            else:
                loss = loss_fn(normed_output, normed_x)
            loss.backward()
            nan_name = check_nan_gradients(model)
            if nan_name is not None:
                logger.error(f'Gradient has NaN value in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] in parameters {nan_name}')
                raise ValueError(f'Gradient has NaN value in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] first in {nan_name} in backward propagation')
            # assert len(nan_list) == 0, f'Gradient has NaN value in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]'
            optimizer.step()
            # recover data
            output = data_normalization.recover_data(normed_output, config['model']['use_one_channel'])

        # recover data
        else:
            output = data_normalization.recover_data(normed_output, config['model']['use_one_channel'])
            # print(output.size())
            if masked_flag:
                loss = loss_fn(output[:, t_in:], x[:, t_in:])
            else:
                loss = loss_fn(output, x)
            loss.backward()
            nan_name = check_nan_gradients(model)
            if nan_name is not None:
                logger.error(f'Gradient has NaN value in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] first in {nan_name} in backward propagation')
                raise ValueError(f'Gradient has NaN value in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] first in {nan_name} in backward propagation')
            # assert len(nan_list) == 0, f'Gradient has NaN value in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]'
            optimizer.step()
        # metrics
        rec_mse += ((x[:, :t_in] - output[:, :t_in]) ** 2).detach().cpu().mean().item()
        # only unknowns
        if masked_flag:
            x, output = x[:, t_in:], output[:, t_in:]

        # loss has nan
        if torch.isnan(loss).any():
            logger.error(f'Loss is NaN in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]')
            raise ValueError(f'Loss is NaN in [Epoch {epoch}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}]')
        # iteration_count += 1
        if args.loggrad != -1:
            log_gradients(epoch, num_epochs, iteration_count, train_loader, model, grad_logger, args)

        # clamp data: >0 clamp to [0, args.clamp], =0 clamp to [0, inf), <0 no clamp
        if config['clamp'] > 0:
            model.clamp_param(config['clamp']) # remove restrictions for x
        elif config['clamp'] == 0:
            model.clamp_param()
        if args.debug:
            torch.save(model.state_dict(), debug_model_path)

        # log RMSE on each step during training, 40 times per epoch
        rmse_per_time += ((x - output) ** 2).detach().cpu().mean((0,2,3)) # in (T,)
        rmse_sep += loss.detach().cpu().item()
        check_per_epoch = 5
        if iteration_count % (len(train_loader) // check_per_epoch) == 0:
            batch_step = epoch * len(train_loader) + iteration_count#  / len(train_loader)
            # print(batch_step)
            rmse_checkpoint = torch.sqrt(rmse_per_time / (len(train_loader) // check_per_epoch))# .cpu().detach()
            # print(rmse_checkpoint.min().item(), rmse_checkpoint.max().item())
            rec_rmse_dict = {f'time_{i:02d}': rmse_checkpoint[i] for i in range(t_in)}
            writer.add_scalars('rec_RMSE_per_step', rec_rmse_dict, global_step=batch_step)
            pred_rmse_dict = {f'time_{i:02d}': rmse_checkpoint[i] for i in range(t_in, T)}
            writer.add_scalars('pred_RMSE_per_step', pred_rmse_dict, global_step=batch_step)
            rmse_dict = {f'time_{i:02d}': rmse_checkpoint[i] for i in range(T)}
            writer.add_scalars('RMSE_per_step', rmse_dict, global_step=batch_step)
            writer.add_scalar('Loss_batch', rmse_sep / (len(train_loader) // check_per_epoch), global_step=batch_step)

            # log parameter scalars
            param_dicts, grad_dict = log_parameters_scalars(model, ['multiQ', 'multiM', 'alpha', 'beta'])
            for check_name, value_dict in param_dicts.items():
                writer.add_scalars(check_name, value_dict, global_step=batch_step)
            writer.add_scalars('grad', grad_dict, global_step=batch_step)
            rmse_per_time = rmse_per_time * 0
            rmse_sep = 0

        # cauculate metrics
        running_loss += loss.item()
        # with torch.no_grad():
        if masked_flag:
            pred_mse += ((x - output) ** 2).detach().cpu().mean().item()
            pred_mae += (torch.abs(output - x)).detach().cpu().mean().item()
            # mape
            mask = (x > 1e-8)
            pred_mape += (torch.abs(output[mask] - x[mask]) / x[mask]).detach().cpu().mean().item() * 100
            nearest_loss += ((x[:, 0] - output[:, 0]) ** 2).detach().cpu().mean().item()
        else:
            x_pred = x[:, t_in:]
            output_pred = output[:, t_in:]
            mask = (x_pred > 1e-8)
            pred_mse += ((x_pred - output_pred) ** 2).detach().cpu().mean().item()
            pred_mae += (torch.abs(output_pred - x_pred)).detach().cpu().mean().item()
            pred_mape += (torch.abs(output_pred[mask] - x_pred[mask]) / x_pred[mask]).detach().cpu().mean().item() * 100
            nearest_loss += ((x[:, t_in] - output[:, t_in]) ** 2 / y.size(0)).detach().cpu().mean().item()

        # glm = model.model_blocks[0]['graph_learning_module']
        # admm_block = model.model_blocks[0]['ADMM_block']
        # break
    
    # log on each epoch
    logger.info('output: (%f, %f)', output.detach().cpu().max().item(), output.detach().cpu().min().item())
    # print_gradients(model)
    total_loss = running_loss / len(train_loader)
    train_loss_list.append(total_loss)
    nearest_rmse = math.sqrt(nearest_loss / len(train_loader))
    rec_rmse = math.sqrt(rec_mse / len(train_loader))
    pred_rmse = math.sqrt(pred_mse / len(train_loader))
    pred_mae = pred_mae / len(train_loader)
    pred_mape = pred_mape / len(train_loader)

    logger.info('Training: Epoch [%d/%d], LR:%.2e, Loss:%.4f, rec_RMSE: %.4f, RMSE_next:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', epoch + 1, num_epochs, optimizer.param_groups[0]['lr'], total_loss, rec_rmse, nearest_rmse, pred_rmse, pred_mae, pred_mape)
    # print other parameters
    # print_parameters(model, ['multiQ', 'alpha', 'beta'], logger)

    # logger.info('multiQ1, multiQ2, multiM: %f, %f, %f', glm.multiQ1.max().item(), glm.multiQ2.max().item(), glm.multiM.max().item())
    # if args.ablation in ['None', 'DGLR']:
    #     logger.info('rho: %f', admm_block.rho.max().item())
    # logger.info('rho_u: %f', admm_block.rho_u.max().item())
    # if args.ablation != 'DGLR':
    #     logger.info('rho_d: %f', admm_block.rho_d.max().item())

    # logger.info('alpha_x, (%.4f, %.4f), beta_x (%.4f, %.4f)', admm_block.alpha_x.min().item(), admm_block.alpha_x.max().item(), admm_block.beta_x.min().item(), admm_block.beta_x.max().item())
    # logger.info('alpha_zu, (%.4f, %.4f), beta_zu (%.4f, %.4f)', admm_block.alpha_zu.min().item(), admm_block.alpha_zu.max().item(), admm_block.beta_zu.min().item(), admm_block.beta_zu.max().item())
    # if args.ablation != 'DGLR':
    #     logger.info('alpha_zd, (%.4f, %.4f), beta_zd (%.4f, %.4f)', admm_block.alpha_zd.min().item(), admm_block.alpha_zd.max().item(), admm_block.beta_zd.min().item(), admm_block.beta_zd.max().item())

    # validation
    # if (epoch + 1) % 5 == 0:
    if not config['model']['use_one_channel']:
        val_loss, metrics, metric_d = test(model, val_loader, data_normalization, masked_flag, config, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=False)
    else:
        val_loss, metrics = test(model, val_loader, data_normalization, masked_flag, config, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=True)

    val_loss_list.append(val_loss)

    logger.info('Validation: Epoch [%d/%d], Loss:%.4f, rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', epoch + 1, num_epochs, val_loss, metrics['rec_RMSE'], metrics['pred_RMSE'], metrics['pred_MAE'], metrics['pred_MAPE'])

    if not config['model']['use_one_channel']:
        for i in range(signal_channels):
            logger.info('Channel %s:\t rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', signal_list[i], metric_d['rec_RMSE'][i], metric_d['pred_RMSE'][i], metric_d['pred_MAE'][i], metric_d['pred_MAPE'][i])
    # save model dicts
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        logger.info(f'saved best params at epoch {epoch + 1}')
        torch.save(model.state_dict(), os.path.join(model_dir, f'val_{epoch+1}.pth'))

    # if val_metrics['']
    # torch.save(model.state_dict(), os.path.join(model_dir, f'val_{epoch+1}.pth'))
    if args.use_stepLR:
        scheduler.step(val_loss)
        logger.info('Current Learning Rate: %.2e', optimizer.param_groups[0]['lr'])
    
    # test
    if (epoch + 1) % 10 == 0 and best_epoch > epoch + 1 - 10:
        gc.collect()
        torch.cuda.empty_cache()
        best_model = copy.deepcopy(model)
        best_model.load_state_dict(torch.load(os.path.join(model_dir, f'val_{best_epoch}.pth'), weights_only=True))
        best_model.zero_grad()
        if not config['model']['use_one_channel']:
            test_loss, test_metrics, test_metric_d = test(best_model, test_loader, data_normalization, masked_flag, config, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=False)
        else:
            test_loss, test_metrics = test(best_model, test_loader, data_normalization, masked_flag, config, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=True)

        logger.info('Test: Epoch [%d/%d/%d], Loss:%.4f, rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', best_epoch, epoch + 1, num_epochs, test_loss, test_metrics['rec_RMSE'], test_metrics['pred_RMSE'], test_metrics['pred_MAE'], test_metrics['pred_MAPE'])

        gc.collect()
        torch.cuda.empty_cache()
        del best_model

plot_loss_curve(train_loss_list, val_loss_list, plot_path)