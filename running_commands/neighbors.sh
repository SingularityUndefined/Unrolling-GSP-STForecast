cuda_device=0
python_path="python"
while getopts "c:p:" opt; do
    case $opt in
        c) cuda_device=$OPTARG ;;
        p) python_path=$OPTARG ;;
        \?) echo "Invalid option -$OPTARG" >&2 ;;
    esac
done



commands=(
    "$python_path train.py --dataset PEMS04 --cuda 1 --batchsize 12 --neighbors 6 --loggrad -1"
    "$python_path train.py --dataset PEMS04 --cuda 1 --batchsize 16 --neighbors 4 --loggrad -1"
    "$python_path train.py --dataset PEMS04 --cuda 1 --batchsize 12 --neighbors 8 --loggrad -1"
    "$python_path train.py --dataset PEMS04 --cuda 1 --batchsize 16 --neighbors 2 --loggrad -1"

)

log_file="running_commands/neighbors_test.log"
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