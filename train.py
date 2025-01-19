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
parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
parser.add_argument('--debug', dest='debug', help='if debug, save model every iteration', action='store_true')
parser.set_defaults(debug=False)
parser.add_argument('--optim', help='optimizer', default='adamw', type=str)
parser.add_argument('--mode', help='normalization mode', default='normalize', type=str)
parser.add_argument('--ablation', dest='ablation', default='None', type=str)#action='store_true', help='run ablation model') 
parser.add_argument('--stepsize', help='stepLR stepsize', default=8, type=int)
parser.add_argument('--gamma', help='stepLR gamma', default=0.2, type=float)
# parser.set_defaults(ablation=False)
# parser.add_argument('--no-flag', dest='flag', action='store_false', help='设置标志为False')
# parser.add_argument('--ablation', help='is abalation model', default=False, type=bool)
parser.add_argument('--loggrad', help='log gradient norms', default=20, type=int)
parser.add_argument('--epochs', help='running epochs', default=30, type=int)

parser.add_argument('--clamp', help='clamp parameter', default=0.20, type=float)
parser.add_argument('--extrapolation', help='use extrapolation', action='store_true', default=False)
parser.set_defaults(extrapolation=False)
args = parser.parse_args()

seed_everything(args.seed)
# Hyper-parameter[s
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
batch_size = args.batchsize
learning_rate = args.lr# 1e-3
num_epochs = args.epochs
num_workers = 4

# load datasets
loss_name = 'MSE'

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

k_hop = args.hop
dataset_dir = '../datasets/PEMS0X_data/'
experiment_name = f'{k_hop}_hop_concatFE_{args.tin}_{args.tout}_seed{args.seed}'
if args.ablation != 'None':
    experiment_name = 'wo_{args.ablation}' + experiment_name
dataset_name = args.dataset
T = args.tin + args.tout
t_in = args.tin
stride = 3

return_time = True

train_set, val_set, test_set, train_loader, val_loader, test_loader = create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time)
signal_channels = train_set.signal_channel

print('signal channels', signal_channels)
# data normalization
data_normalization = Normalization(train_set, args.mode)

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

model_pretrained_path = None


print('args.ablation', args.ablation)
model = UnrollingModel(num_admm_blocks, device, T, t_in, num_heads, train_set.signal_channel, feature_channels, GNN_layers=2, graph_info=train_set.graph_info, ADMM_info=ADMM_info, k_hop=k_hop, ablation=args.ablation, use_extrapolation=args.extrapolation).to(device)
# 'UnrollingForecasting/MainExperiments/models/v2/PEMS04/direct_4b_4h_6f/val_15.pth'

if model_pretrained_path is not None:
    model = model.load_state_dict(torch.load(model_pretrained_path))

ADMM_iters = ADMM_info['ADMM_iters']
# optimizer
import torch.optim as optim
from torch.optim import lr_scheduler

assert args.optim in ['adam', 'adamw'], 'args.optim should be adam or adamw'
if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma) # TODO: step size

# 创建文件处理器
log_dir = f'../JanModified/logs_midparam/{experiment_name}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{dataset_name}_{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.log'

logger = setup_logger('logger1', os.path.join(log_dir, log_filename), logging.DEBUG, to_console=True)
if args.loggrad:
    grad_logger_dir = f'../JanModified/grad_logs_midparam/{experiment_name}'
    os.makedirs(grad_logger_dir, exist_ok=True)
    grad_logger = setup_logger('logger2', os.path.join(grad_logger_dir, log_filename), logging.INFO, to_console=False)

debug_model_path = os.path.join(f'../JanModified/debug_models_midparam/{experiment_name}', f'{dataset_name}/{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.pth')

print('log dir', log_dir)
logger.info('#################################################')
logger.info('PARAMETER SETTINGS:')
for arg, value in vars(args).items():
    logger.info(f"{arg}: {value}")
logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'pretrained path: {model_pretrained_path}')
# logger.info(f'learning k hop: {k_hop}')
logger.info(f'feature channels: {feature_channels}')
# logger.info(f'graph sigma: {graph_sigma}')
# logger.info(f'batch size: {batch_size}')
# logger.info(f'learning rate: {learning_rate}')
logger.info(f'Loss function: {loss_name}')
logger.info(f"Total parameters: {total_params}")
# logger.info(f'device: {device}')
logger.info('PARAMTER SETTINGS:')
logger.info(f'ADMM blocks: {num_admm_blocks}')
logger.info(f'ADMM info: {ADMM_info}')
logger.info(f'graph info: nodes {train_set.n_nodes}, edges {train_set.n_edges}, signal channels {signal_channels}')
logger.info('--------BEGIN TRAINING PROCESS------------')

grad_logger.info('------BEGIN TRAINING PROCESS-------')

model_dir = os.path.join(f'../JanModified/models_midparam/{experiment_name}', f'{dataset_name}/{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.pth')
os.makedirs(model_dir, exist_ok=True)
masked_flag = False
# train models
# test = True
plot_list = f'../JanModified/loss_curve_midparam/{experiment_name}'
os.makedirs(plot_list, exist_ok=True)
plot_filename = f'{dataset_name}_{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.png'
plot_path = os.path.join(plot_list, plot_filename)

train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    # TODO: remember to / 50 # don't need now
    model.train()
    running_loss = 0
    rec_mse = 0
    pred_mse = 0
    pred_mae = 0
    pred_mape = 0
    nearest_loss = 0

    iteration_count = 0

    for y, x, t_list in tqdm(train_loader):
        # print(y.shape, x.shape)
        optimizer.zero_grad()
        y, x, t_list = y.to(device), x.to(device), t_list.to(device) # y in (B, t, nodes, 1)
        # normalization
        # y = (y - train_mean) / train_std
        y = data_normalization.normalize_data(y)
        # y = (y - train_min) / (train_max - train_min)
        try:
            output = model(y, t_list) # in (B, T, nodes, 1)
        except ValueError as ve:
            # plot the loss curve from loss_list
            # plot_loss_curve(loss_list, log_dir)
            plot_loss_curve(train_loss_list, val_loss_list, plot_path)
            logger.error(f'Error in [Epoch {epoch}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] - {ve}')
            grad_logger.error(f'Error in [Epoch {epoch}/{num_epochs}, Iter {iteration_count}/{len(train_loader)}] - {ve}')
        # except AssertionError as ae:
           #  print(f'Error in [Epoch {epoch}, Iter {iteration_count}] - {ae}')
        output = data_normalization.recover_data(output)
        if args.mode == 'normalize':
        # output = output * (train_max - train_min) + train_min # output * train_std + train_mean
            output = nn.ReLU()(output)# [:,:,:,:signal_channels]

        rec_mse += ((x[:,:t_in] - output[:,:t_in]) ** 2).mean().item()
        # only unknowns
        if masked_flag:
            x, output = x[:,t_in:], output[:,t_in:]
        loss = loss_fn(output, x)
        loss.backward()       
        optimizer.step()
        iteration_count += 1

        log_gradients(epoch, num_epochs, iteration_count, train_loader, model, grad_logger, args)
        
        if args.clamp > 0:
            model.clamp_param(args.clamp, args.clamp)
        if args.debug:
            torch.save(model.state_dict(), debug_model_path)
        # loggers
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
            x_pred = x[:,t_in:]
            output_pred = output[:,t_in:]   
            mask = (x_pred > 1e-8)
            pred_mse += ((x_pred - output_pred) ** 2).mean().item()
            pred_mae += (torch.abs(output_pred - x_pred)).mean().item()
            pred_mape += (torch.abs(output_pred[mask] - x_pred[mask]) / x_pred[mask]).mean().item() * 100
            nearest_loss += ((x[:, t_in] - output[:, t_in]) ** 2 / y.size(0)).mean().item()

        glm = model.model_blocks[0]['graph_learning_module']
        admm_block = model.model_blocks[0]['ADMM_block']
        # break
    
    logger.info(f'output: ({output.max().item()}, {output.min().item()})')

    total_loss = running_loss / len(train_loader)
    train_loss_list.append(total_loss)
    nearest_rmse = math.sqrt(nearest_loss / len(train_loader))
    rec_rmse = math.sqrt(rec_mse / len(train_loader))
    pred_rmse = math.sqrt(pred_mse / len(train_loader))
    pred_mae = pred_mae / len(train_loader)
    pred_mape = pred_mape / len(train_loader)
    metrics = {
        'rec_RMSE': rec_rmse,
        'pred_RMSE': pred_rmse,
        'pred_MAE': pred_mae,
        'pred_MAPE(%)': pred_mape 
    }

    logger.info(f'Training: Epoch [{epoch + 1}/{num_epochs}], Loss:{total_loss:.4f}, rec_RMSE: {rec_rmse:.4f}, RMSE_next:{nearest_rmse:.4f}, RMSE:{pred_rmse:.4f}, MAE:{pred_mae:.4f}, MAPE(%):{pred_mape:.4f}')
    logger.info(f'multiQ1, multiQ2, multiM: {glm.multiQ1.max().item()}, {glm.multiQ2.max().item()}, {glm.multiM.max().item()}')
    if not args.ablation:
        logger.info(f'rho: {admm_block.rho.max().item()}')
    logger.info(f'rho_u, rho_d: {admm_block.rho_u.max().item()}, {admm_block.rho_d.max().item()}')
    logger.info(f'max alphas, {admm_block.alpha_x.max().item():.4f}, {admm_block.alpha_zu.max().item():.4f}, {admm_block.alpha_zd.max().item():.4f}')
    logger.info(f'min alphas, {admm_block.alpha_x.min().item():.4f}, {admm_block.alpha_zu.min().item():.4f}, {admm_block.alpha_zd.min().item():.4f}')
    logger.info(f'max betas, {admm_block.beta_x.max().item():.4f}, {admm_block.beta_zu.max().item():.4f}, {admm_block.beta_zd.max().item():.4f}')
    logger.info(f'min betas, {admm_block.beta_x.min().item():.4f}, {admm_block.beta_zu.min().item():.4f}, {admm_block.beta_zd.min().item():.4f}')

    # validation
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            running_loss = 0
            nearest_loss = 0
            rec_mse = 0
            pred_mse = 0
            pred_mape = 0
            pred_mae = 0
            for y, x, t_list in tqdm(val_loader):
                y, x, t_list = y.to(device), x.to(device), t_list.to(device)

                # y = (y - train_mean) / train_std
                # y = (y - train_min) / (train_max - train_min)
                y = data_normalization.normalize_data(y)
                output = model(y, t_list)
                output = data_normalization.recover_data(output)
                if args.mode == 'normalize':
                    output = nn.ReLU()(output)
                # output = output * (train_max - train_min) + train_min
                # output = output * train_std + train_mean

                rec_mse += ((x[:,:t_in] - output[:,:t_in]) ** 2).mean().item()
                if masked_flag:
                    x, output = x[:,t_in:], output[:,t_in:]
                loss = loss_fn(output, x)

                running_loss += loss.item()
                if masked_flag:
                    pred_mse += ((x - output) ** 2).mean().item()
                    pred_mae += (torch.abs(output - x)).mean().item()
                    mask = (x > 1e-8)
                    pred_mape += (torch.abs(output[mask] - x[mask]) / x[mask]).mean().item() * 100
                    nearest_loss += ((x[:, 0] - output[:, 0]) ** 2).mean().item()
                else:
                    x_pred = x[:,t_in:]
                    output_pred = output[:,t_in:]
                    mask = x_pred > 1e-8
                    pred_mse += ((x_pred - output_pred) ** 2).mean().item()
                    pred_mae += (torch.abs(output_pred - x_pred)).mean().item()
                    pred_mape += (torch.abs(output_pred[mask] - x_pred[mask]) / x_pred[mask]).mean().item() * 100
                    nearest_loss += ((x[:, t_in] - output[:, t_in]) ** 2).mean().item()

        total_loss = running_loss / len(val_loader)
        val_loss_list.append(total_loss)
        nearest_rmse = math.sqrt(nearest_loss / len(val_loader))
        rec_rmse = math.sqrt(rec_mse / len(val_loader))
        pred_rmse = math.sqrt(pred_mse / len(val_loader))
        pred_mae = pred_mae / len(val_loader)
        pred_mape = pred_mape / len(val_loader)
        metrics = {
            'rec_RMSE': rec_rmse,
            'pred_RMSE': pred_rmse,
            'pred_MAE': pred_mae,
            'pred_MAPE(%)': pred_mape 
        }
        
        logger.info(f'Validation: Epoch [{epoch + 1}/{num_epochs}], Loss:{total_loss:.4f}, rec_RMSE:{rec_rmse:.4f} RMSE_next:{nearest_rmse:.4f}, RMSE:{pred_rmse:.4f}, MAE:{pred_mae:.4f}, MAPE(%):{pred_mape:.4f}')
        # save models
        torch.save(model, os.path.join(model_dir, f'val_{epoch+1}.pth'))

    # rmse_total = math.sqrt(avg_mse_loss)
plot_loss_curve(train_loss_list, val_loss_list, plot_path)