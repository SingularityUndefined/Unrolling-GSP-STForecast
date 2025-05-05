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
    "$python_path train_weather.py --dataset Molene --batchsize 64 --cuda $cuda_device --neighbors 6 --interval 6 --epoch 200 --stepLR --stepsize 60"
    "$python_path train_weather.py --dataset Molene --batchsize 64 --cuda $cuda_device --neighbors 10 --interval 6 --epoch 200 --stepLR --stepsize 60"
    "$python_path train_weather.py --dataset Molene --batchsize 64 --cuda $cuda_device --neighbors 10 --interval 3 --epoch 200 --stepLR --stepsize 60"
    "$python_path train_weather.py --dataset Molene --batchsize 64 --cuda $cuda_device --neighbors 6 --interval 5 --epoch 200 --stepLR --stepsize 60"

)

log_file="running_commands/Molene.log"
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