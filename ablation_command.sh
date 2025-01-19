#!/bin/bash

# 定义要执行的命令
commands=(
    "python train.py --dataset PEMS03 --hop 4 --batchsize 12  --loggrad 10 --cuda 0 --lr 1e-4  --stepsize 8 --epoch 50 --seed 42 --ablation"
    "python train.py --dataset PEMS04 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-4  --stepsize 8 --epoch 50 --seed 42 --ablation"
    # "python train.py --dataset PEMS07 --hop 4 --batchsize 6  --loggrad 10 --cuda 0 --lr 1e-4  --stepsize 8 --epoch 50 --seed 42 --ablation"
    "python train.py --dataset PEMS08 --hop 4 --batchsize 16 --loggrad 10 --cuda 0 --lr 1e-4 --stepsize 8 --epoch 50 --seed 42 --ablation"
)

# 日志文件
log_file="ablation_command_execution.log"

# 遍历命令并执行
for cmd in "${commands[@]}"; do
    echo "Executing: $cmd"
    start_time=$(date +%s)
    if $cmd; then
        status="Success"
    else
        status="Failed"
    fi
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Command: $cmd" >> $log_file
    echo "Status: $status" >> $log_file
    echo "Duration: ${duration}s" >> $log_file
    echo "-----------------------------" >> $log_file
done