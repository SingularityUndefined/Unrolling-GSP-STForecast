#!/bin/bash
# 安装所需的Python包
#!/bin/bash
# 安装所需的Python包
# required_packages=(torch tqdm numpy matplotlib networkx pandas)
# for package in "${required_packages[@]}"; do
#     if ! $python_path -c "import $package" &> /dev/null; then
#         pip install $package
#     fi
# done
cuda_device=0
python_path="python"
while getopts "c:p:" opt; do
    case $opt in
        c) cuda_device=$OPTARG ;;
        p) python_path=$OPTARG ;;
        \?) echo "Invalid option -$OPTARG" >&2 ;;
    esac
done
# TODO: Update the batchsize and loggrad parameters
# 定义要执行的命令
commands=(
    # todo: Needs to lower learning rate for main experiments
    # "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 16  --loggrad 20 --cuda $cuda_device --lr 1e-3 --clamp 0.8 --extrapolation --loss Huber --flow --diffM"
    # "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 28  --loggrad 20 --cuda $cuda_device --lr 1e-3 --clamp 0.8 --extrapolation --loss Huber --ablation DGTV --flow"
    # "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 24  --loggrad 20 --cuda $cuda_device --lr 1e-3 --clamp 0.8 --extrapolation --loss Huber --ablation DGLR"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 16  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --flow --diffM"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 28  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGTV --flow --diffM"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 24  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGLR --flow --diffM"

    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 16  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --flow --diffM"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 28  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGTV --flow --diffM"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 24  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGLR --flow --diffM"

    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 16  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss MSE --flow --diffM"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 28  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss MSE --ablation DGTV --flow --diffM"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 24  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss MSE --ablation DGLR --flow --diffM"

    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 6  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --diffM"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 6  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGTV --diffM"
    "$python_path train.py --dataset PEMS08 --hop 4 --batchsize 8  --loggrad 20 --cuda $cuda_device --lr 1e-4 --clamp 0.8 --extrapolation --loss Huber --ablation DGLR --diffM"
)

# 日志文件
log_file="execution_08.log"

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