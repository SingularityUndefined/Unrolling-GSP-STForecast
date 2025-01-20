#!/bin/bash
# 安装所需的Python包
pip install torch tqdm numpy matplotlib networkx pandas
# 解析命令行参数
cuda_device=0
python_path="python"
while getopts "c:p:" opt; do
    case $opt in
        c) cuda_device=$OPTARG ;;
        p) python_path=$OPTARG ;;
        \?) echo "Invalid option -$OPTARG" >&2 ;;
    esac
done

# 更新commands中的cuda参数
commands=(
    # lr = 1e-3
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 4  --loggrad 20 --cuda $cuda_device --lr 1e-3 --clamp 0.8 --extrapolation --loss Huber"
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 8  --loggrad 10 --cuda $cuda_device --lr 1e-3 --clamp 0.8 --extrapolation --loss Huber --ablation DGTV"
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 6  --loggrad 10 --cuda $cuda_device --lr 1e-3 --clamp 0.8 --extrapolation --loss Huber --ablation DGLR"
    # lr = 5e-4
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 4  --loggrad 20 --cuda $cuda_device --lr 5e-4 --clamp 0.8 --extrapolation --loss Huber"
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 8  --loggrad 10 --cuda $cuda_device --lr 5e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGTV"
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 6  --loggrad 10 --cuda $cuda_device --lr 5e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGLR"
    # lr = 1e-4
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 4  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber"
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 8  --loggrad 10 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGTV"
    "$python_path train.py --dataset PEMS07 --hop 4 --batchsize 6  --loggrad 10 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGLR"
)

# 日志文件
log_file="execution_07.log"

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