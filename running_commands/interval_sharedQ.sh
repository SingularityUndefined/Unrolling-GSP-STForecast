cuda_device=0
seed=42
python_path="python"
while getopts "c:p:" opt; do
    case $opt in
        c) cuda_device=$OPTARG ;;
        p) python_path=$OPTARG ;;
        \?) echo "Invalid option -$OPTARG" >&2 ;;
    esac
done



commands=(
    "$python_path train.py --dataset PEMS04 --cuda $cuda_device --batchsize 14 --interval 4 --loggrad -1 --seed $seed --sharedQ"
    "$python_path train.py --dataset PEMS04 --cuda $cuda_device --batchsize 12 --interval 6 --loggrad -1 --seed $seed --sharedQ"
    "$python_path train.py --dataset PEMS04 --cuda $cuda_device --batchsize 16 --interval 2 --loggrad -1 --seed $seed --sharedQ"

)

log_file="running_commands/interval_diffM.log"
echo "--------------NEW RUN-------------" >> $log_file

for cmd in "${commands[@]}"; do
    echo "Executing Command: $cmd" >> $log_file
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Start Time: $start_time" >> $log_file
    if $cmd; then
        status="Success"
    else
        status="Failed"
    fi
    end_time=$(date '+%Y-%m-%d %H:%M:%S')
    # echo "Command: $cmd" >> $log_file
    echo "Status: $status" >> $log_file
    # echo "Start Time: $start_time" >> $log_file
    echo "End Time: $end_time" >> $log_file
    echo "------" >> $log_file
done