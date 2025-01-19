#!/bin/bash
# 安装所需的Python包
pip install torch tqdm numpy matplotlib networkx pandas
# 定义要执行的命令
commands=(
    # "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-3  --stepsize 10 --epoch 50 --optim adam"
    "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-3  --stepsize 10 --epoch 50 --clamp 0.5 --extrapolation"
    "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-3  --stepsize 10 --epoch 50 --clamp 0.8 --extrapolation"
    "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-3  --stepsize 10 --epoch 50 --clamp -1   --extrapolation"
    "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-3  --stepsize 5 --epoch 50 --clamp 0.5 --extrapolation"
    "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-3  --stepsize 10 --epoch 50 --clamp 0.5 --optim adam --extrapolation"
    # "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 5e-4  --stepsize 10 --epoch 50 --"
    # "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-4  --stepsize 10 --epoch 50"
    # "python train.py --dataset PEMS07 --hop 4 --batchsize 4  --loggrad 10 --cuda 0 --lr 1e-4  --stepsize 8 --epoch 50 --seed 42"
    # "python train.py --dataset PEMS08 --hop 4 --batchsize 10 --loggrad 10 --cuda 0 --lr 1e-4 --stepsize 8 --epoch 50 --seed 42"
)

# 日志文件
log_file="train_command_execution.log"

# 遍历命令并执行
for cmd in "${commands[@]}"; do
    echo "Executing: $cmd"
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    if $cmd; then
        status="Success"
    else
        status="Failed"
    fi
    end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Command: $cmd" >> $log_file
    echo "Status: $status" >> $log_file
    echo "Start Time: $start_time" >> $log_file
    echo "End Time: $end_time" >> $log_file
    echo "-----------------------------" >> $log_file
done