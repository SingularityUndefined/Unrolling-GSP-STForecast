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

parser = argparse.ArgumentParser()

# device and data
parser.add_argument('--cuda', help='CUDA device', type=int)
parser.add_argument('--dataset', help='dataset name', type=str)
parser.add_argument('--batchsize', help='batch size', type=int)
parser.add_argument('--mode', help='normalization mode', default='normalize', type=str)

# experiment type
parser.add_argument('--ablation', help='operator to elimnate in ablation study', default='None', type=str)#action='store_true', help='run ablation model')

# model hyper-params
parser.add_argument('--tin', help='time input', default=12, type=int)
parser.add_argument('--tout', help='time output', default=12, type=int)
parser.add_argument('--hop', help='k for kNN', default=6, type=int)
parser.add_argument('--numblock', help='number of admm blocks', default=5, type=int)
parser.add_argument('--numlayer', help='number of admm layers', default=25, type=int)
parser.add_argument('--cgiter', help='CGD iterations', default=3, type=int)
parser.add_argument('--seed', help='random seed', default=3407, type=int)
parser.add_argument('--debug', dest='debug', help='if debug, save model every iteration', action='store_true')
parser.set_defaults(debug=False)

# optimizer and learning setting
parser.add_argument('--optim', help='optimizer', default='adamw', type=str)
parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
parser.add_argument('--stepLR', dest='use_stepLR', action='store_true')
parser.set_defaults(use_stepLR=False)
parser.add_argument('--stepsize', help='stepLR stepsize', default=8, type=int)
parser.add_argument('--gamma', help='stepLR gamma', default=0.2, type=float)

# training settings
parser.add_argument('--epochs', help='running epochs', default=30, type=int)
parser.add_argument('--clamp', help='clamp parameter', default=0.20, type=float) # clamp CG params
parser.add_argument('--loss', help='loss function', default='MSE', type=str)

# log settings
parser.add_argument('--loggrad', help='log gradient norms', default=20, type=int) # -1 stands for no log

# default settings, extrapolation model
parser.add_argument('--extrapolation', help='use extrapolation GCN model', action='store_true')#, default=False)
parser.set_defaults(extrapolation=False)

# default settings, 1-channel prediction
parser.add_argument('--flow', help='flow channel', dest='use_one_channel', action='store_true')
parser.set_defaults(use_one_channel=False)

# default settings, no shared parameters
parser.add_argument('--diffM', dest='shared_params', action='store_false')
parser.set_defaults(shared_params=True)

# default settings, normalized loss
parser.add_argument('--normloss', dest='normed_loss', action='store_true')
parser.set_defaults(normed_loss=False)

args = parser.parse_args()

# seed, device, training settings
seed_everything(args.seed)
# Hyper-parameter[s
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
batch_size = args.batchsize
learning_rate = args.lr# 1e-3
num_epochs = args.epochs
num_workers = 4

# loss
loss_name = args.loss

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
k_hop = args.hop
dataset_dir = '/home/disk/qij/TS_datasets/PEMS0X_data/'
experiment_name = f'{k_hop}_hop_lr_{args.lr:.0e}_seed{args.seed}'

if args.ablation != 'None':
    experiment_name = f'wo_{args.ablation}' + experiment_name
if not args.extrapolation:
    experiment_name = 'LR_' + experiment_name

if args.use_one_channel:
    experiment_name = '1channel_' + experiment_name

if not args.shared_params:
    experiment_name = 'diffM_' + experiment_name

if args.normed_loss:
    experiment_name = experiment_name + '_normed_loss'
else:
    experiment_name = experiment_name + '_true_loss'

dataset_name = args.dataset
T = args.tin + args.tout
t_in = args.tin
stride = 3

return_time = True

# load data
train_set, val_set, test_set, train_loader, val_loader, test_loader = create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time, use_one_channel=args.use_one_channel) # use one channel
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

num_admm_blocks = args.numblock
num_heads = 4
interval = 4
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

model_pretrained_path = None


print('args.ablation', args.ablation)
model = UnrollingModel(num_admm_blocks, device, T, t_in, num_heads, interval, train_set.signal_channel, feature_channels, GNN_layers=2, graph_info=train_set.graph_info, ADMM_info=ADMM_info, k_hop=k_hop, ablation=args.ablation, use_extrapolation=args.extrapolation, use_one_channel=args.use_one_channel, shared_params=args.shared_params).to(device)
# 'UnrollingForecasting/MainExperiments/models/v2/PEMS04/direct_4b_4h_6f/val_15.pth'

if model_pretrained_path is not None:
    model = model.load_state_dict(torch.load(model_pretrained_path))
    # TODO: map to models

ADMM_iters = ADMM_info['ADMM_iters']

# optimizer
import torch.optim as optim
from torch.optim import lr_scheduler

assert args.optim in ['adam', 'adamw'], 'args.optim should be adam or adamw'
if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

if args.use_stepLR:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma) # TODO: step size

# tensorboard logger
tensorboard_logdir = f'./dense_logs/TB_log/{experiment_name}'
os.makedirs(tensorboard_logdir, exist_ok=True)
writer = SummaryWriter(tensorboard_logdir)

# create loggers
log_dir = f'./dense_logs/train_Logs/{experiment_name}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{dataset_name}_{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.log'

logger = setup_logger('logger1', os.path.join(log_dir, log_filename), logging.DEBUG, to_console=True)

if args.loggrad != -1:
    grad_logger_dir = f'./dense_logs/grad_logs/{experiment_name}'
    os.makedirs(grad_logger_dir, exist_ok=True)
    grad_logger = setup_logger('logger2', os.path.join(grad_logger_dir, log_filename), logging.INFO, to_console=False)

# model save dir
debug_model_path = os.path.join(f'./dense_logs/debug_models/{experiment_name}', f'{dataset_name}/{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.pth')
model_dir = os.path.join(f'./dense_logs/models/{experiment_name}', f'{dataset_name}/{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.pth')
os.makedirs(model_dir, exist_ok=True)

# print('log dir', log_dir)
logger.info('#################################################')
logger.info("Training CMD:\t"+ " ".join(sys.argv))
logger.info('PARAMETER SETTINGS:')
for arg, value in vars(args).items():
    logger.info("\t %s: %s", arg, value)
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
print('log dir', log_dir)
masked_flag = False
# train models
# test = True
plot_list = f'./dense_logs/loss_curves/{experiment_name}'
os.makedirs(plot_list, exist_ok=True)
plot_filename = f'{dataset_name}_{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.png'
plot_path = os.path.join(plot_list, plot_filename)

train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    # TODO: remember to / 50 # don't need now
    model.train()
    running_loss = 0
    rmse_per_time = torch.zeros((T,)).to(device)
    rmse_sep = 0
    rec_mse = 0
    pred_mse = 0
    pred_mae = 0
    pred_mape = 0
    nearest_loss = 0

    # iteration_count = 0

    for iter_idx, (y, x, t_list) in enumerate(tqdm(train_loader)):
        # print(y.size(), x.size(), t_list.size())
        iteration_count = iter_idx + 1
        optimizer.zero_grad()
        y, x, t_list = y.to(device), x.to(device), t_list.to(device)  # y in (B, t, nodes, full_signal_channels), x in (B, T, N, pred_signal_channels), T in (B, T)
        # normalization

        normed_y = data_normalization.normalize_data(y)
        normed_x = data_normalization.normalize_data(x, args.use_one_channel)

        try:
            normed_output = model(normed_y, t_list)  # in (B, T, nodes, 1)
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

            if not args.use_one_channel:
                metrics, metrics_d = test(model, test_loader, data_normalization, False, args, device, signal_channels, use_one_channel=False)
            else:
                metrics = test(model, test_loader, data_normalization, False, args, device, signal_channels, use_one_channel=True)

            logger.info('Test (ALL): rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', metrics['rec_RMSE'], metrics['pred_RMSE'], metrics['pred_MAE'], metrics['pred_MAPE'])

            if not args.use_one_channel:
                for i in range(signal_channels):
                    logger.info('Test (%s): rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', signal_list[i], metrics_d['rec_RMSE'][i], metrics_d['pred_RMSE'][i], metrics_d['pred_MAE'][i], metrics_d['pred_MAPE'][i])

            raise ValueError(f'Error in [Epoch {epoch+1}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] - {ve}') from ve

        # normalized loss
        if args.normed_loss:
            if masked_flag:
                loss = loss_fn(normed_output[:, t_in:], normed_x[:, t_in:])
            else:
                loss = loss_fn(normed_output, normed_x)
            loss.backward()
            optimizer.step()
            # recover data
            output = data_normalization.recover_data(normed_output, args.use_one_channel)

        # recover data
        else:
            output = data_normalization.recover_data(normed_output, args.use_one_channel)
            # print(output.size())
            if masked_flag:
                loss = loss_fn(output[:, t_in:], x[:, t_in:])
            else:
                loss = loss_fn(output, x)
            loss.backward()
            optimizer.step()
        
        # Print gradients for all parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                logger.info(f'Parameter {name}: grad_min={param.grad.min().item():.4f}, grad_max={param.grad.max().item():.4f}, grad_mean={param.grad.mean().item():.4f}')
        # metrics
        rec_mse += ((x[:, :t_in] - output[:, :t_in]) ** 2).mean().item()
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
        if args.clamp > 0:
            model.clamp_param(args.clamp, args.clamp)
        elif args.clamp == 0:
            model.clamp_param()
        if args.debug:
            torch.save(model.state_dict(), debug_model_path)

        # log RMSE on each step during training, 40 times per epoch
        rmse_per_time += ((x - output) ** 2).mean((0,2,3)) # in (T,)
        rmse_sep += loss.item()
        check_per_batch = 10
        if iteration_count % (len(train_loader) // check_per_batch) == 0:
            rmse_checkpoint = torch.sqrt(rmse_per_time / (len(train_loader) // 40)).cpu().detach()
            # print(rmse_checkpoint.min().item(), rmse_checkpoint.max().item())
            rec_rmse_dict = {f'time_{i:02d}': rmse_checkpoint[i] for i in range(t_in)}
            writer.add_scalars('rec_RMSE_per_step', rec_rmse_dict, epoch + iteration_count / len(train_loader))
            pred_rmse_dict = {f'time_{i:02d}': rmse_checkpoint[i] for i in range(t_in, T)}
            writer.add_scalars('pred_RMSE_per_step', pred_rmse_dict, epoch + iteration_count / len(train_loader))
            rmse_dict = {f'time_{i:02d}': rmse_checkpoint[i] for i in range(T)}
            writer.add_scalars('RMSE_per_step', rmse_dict, epoch + iteration_count / len(train_loader))
            writer.add_scalar('Loss_batch', rmse_sep / (len(train_loader) // check_per_batch), epoch + iteration_count / len(train_loader))
            rmse_per_time = rmse_per_time * 0
            rmse_sep = 0

        # cauculate metrics
        running_loss += loss.item()
        # with torch.no_grad():
        if masked_flag:
            pred_mse += ((x - output) ** 2).mean().item()
            pred_mae += (torch.abs(output - x)).mean().item()
            # mape
            mask = (x > 1e-8)
            pred_mape += (torch.abs(output[mask] - x[mask]) / x[mask]).mean().item() * 100
            nearest_loss += ((x[:, 0] - output[:, 0]) ** 2).mean().item()
        else:
            x_pred = x[:, t_in:]
            output_pred = output[:, t_in:]
            mask = (x_pred > 1e-8)
            pred_mse += ((x_pred - output_pred) ** 2).mean().item()
            pred_mae += (torch.abs(output_pred - x_pred)).mean().item()
            pred_mape += (torch.abs(output_pred[mask] - x_pred[mask]) / x_pred[mask]).mean().item() * 100
            nearest_loss += ((x[:, t_in] - output[:, t_in]) ** 2 / y.size(0)).mean().item()

        # glm = model.model_blocks[0]['graph_learning_module']
        # admm_block = model.model_blocks[0]['ADMM_block']
        # break
    
    # log on each epoch
    logger.info('output: (%f, %f)', output.max().item(), output.min().item())
    # print_gradients(model)
    total_loss = running_loss / len(train_loader)
    train_loss_list.append(total_loss)
    nearest_rmse = math.sqrt(nearest_loss / len(train_loader))
    rec_rmse = math.sqrt(rec_mse / len(train_loader))
    pred_rmse = math.sqrt(pred_mse / len(train_loader))
    pred_mae = pred_mae / len(train_loader)
    pred_mape = pred_mape / len(train_loader)

    logger.info('Training: Epoch [%d/%d], Loss:%.4f, rec_RMSE: %.4f, RMSE_next:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', epoch + 1, num_epochs, total_loss, rec_rmse, nearest_rmse, pred_rmse, pred_mae, pred_mape)
    # print other parameters

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
    if (epoch + 1) % 5 == 0:
        if not args.use_one_channel:
            running_loss, metrics, metric_d = test(model, val_loader, data_normalization, masked_flag, args, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=False)
        else:
            running_loss, metrics = test(model, val_loader, data_normalization, masked_flag, args, device, signal_channels, mode='val', loss_fn=loss_fn, use_one_channel=True)

        val_loss_list.append(running_loss)

        logger.info('Validation: Epoch [%d/%d], Loss:%.4f, rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', epoch + 1, num_epochs, running_loss, metrics['rec_RMSE'], metrics['pred_RMSE'], metrics['pred_MAE'], metrics['pred_MAPE'])

        if not args.use_one_channel:
            for i in range(signal_channels):
                logger.info('Channel %s:\t rec_RMSE:%.4f, RMSE:%.4f, MAE:%.4f, MAPE(%%):%.4f', signal_list[i], metric_d['rec_RMSE'][i], metric_d['pred_RMSE'][i], metric_d['pred_MAE'][i], metric_d['pred_MAPE'][i])
        # save model dicts
        torch.save(model.state_dict(), os.path.join(model_dir, f'val_{epoch+1}.pth'))

plot_loss_curve(train_loss_list, val_loss_list, plot_path)