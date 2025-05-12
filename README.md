# Lightweight Transformer via Unrolling of Mixed Graph Algorithms for Traffic Forecast

This is a PyTorch implementation of our submission (ID: 4690) to NeurIPS 2025.
## Requirements

Required packages for this implementation:

```
torch>=2.4.1
tqdm 
numpy 
matplotlib 
networkx>=2.5
pandas
tensorboardX
yaml
```
run `pip install -r requirements.txt`.

## Datasets
**PEMS0X datasets** are from repository [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data).

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
```
**PEMS-BAY** and **METR-LA** dataset are from repository []. Each folder contains two `.npy` files for adjacency matrix and time series data. 


## Training and Testing

The default settings are in `config.yaml`. We provide multiple parsers to change the configurations. 

**Example 1**: running main experiment on PEMS03 dataset:
```
python train_traffic.py --dataset PEMS03 --cuda 0 --batchsize 12
```

**Example 2**: running 'w/o DGLR' experiment on METR-LA dataset:
```
python train_traffic.py --dataset METR-LA --cuda 1 --ablation DGLR --batchsize 16
```

**Example 3**: testing a UT model on PEMS-BAY:
```
python test_traffic.py --dataset PEMS-BAY --cuda 0 --ablation UT --batchsize 64 --path <model_checkpoints>
```


