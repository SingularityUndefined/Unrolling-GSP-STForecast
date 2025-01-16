#!/bin/bash

# 定义要执行的命令
commands=(
    "python train.py --dataset PEMS03 --hop 4 --batchsize 10  --loggrad 10 --cuda 0 --lr 1e-4  --stepsize 8 --epoch 50 --seed 42"
    "python train.py --dataset PEMS04 --hop 4 --batchsize 4  --loggrad 10 --cuda 0 --lr 1e-4  --stepsize 8 --epoch 50 --seed 42"
    # "python train.py --dataset PEMS07 --hop 4 --batchsize 4  --loggrad 10 --cuda 0 --lr 1e-4  --stepsize 8 --epoch 50 --seed 42"
    "python train.py --dataset PEMS08 --hop 4 --batchsize 6 --loggrad 10 --cuda 0 --lr 1e-4 --stepsize 8 --epoch 50 --seed 42"
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