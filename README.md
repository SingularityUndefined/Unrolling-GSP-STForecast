# Lightweight Transformer via Unrolling of Mixed Graph Algorithms for Traffic Forecast

This is a PyTorch implementation of our submission (ID: 1707) to ICML 2025.
## Requirements

Required packages for this implementation:

```
torch>=2.4.1
tqdm
numpy 
matplotlib 
networkx>=2.5
pandas
```
run `pip install -r requirements.txt`.

## Datasets
PEMS0X datasets are from repository [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data).

<!-- PEMS-BAY and METR-LA datasets are from repository [DCRNN](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph). -->

Save all the datasets in folder `datasets` parallel to our code folder `Unrolling-GSP-STForecast`. 

Data directories:
```
datasets/
├── PEMS0X——data/
│   ├── PEMS03/
│   ├── PEMS04/
│   ├── PEMS07/
│   ├── PEMS08/
|
```
## Training and Validating
training commands for PEMS0X datasets are in `train_command_0X.sh` (change the 'X' to ['3','4','7','8']) when running the bash files. 
```
bash train_command_0X.sh -c <your-cuda-device> -p <your-python-path>
```
Example running command for PEMS04:
```
python train.py --dataset PEMS04 --hop 4 --batchsize 8  --loggrad 20 --cuda 0 --lr 1e-3 --clamp 0.8 --extrapolation --loss Huber
```


