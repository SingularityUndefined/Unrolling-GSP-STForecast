# Discussion on Dec 5

## Advice on graph
1. Look into the connections of the real edges
    - Thuc: most nodes are under degree 4, while a large number of nodes are of degree 2
    - My suggestion: use a *Dijkstra Algorithm* to find the shortest K paths for each nodes and define connections

2. Feature extractor: max degree 4, just concatenate all the features as an embedding $[{x_j}]$, start with basic models

3. GCN extraction: (1) use multi-layer or multi-connection, choose multilayer; (2) use ReLU as activation

4. Fully directed graph ($A\rightarrow B$ and $B\rightarrow A$ is different)


## Advice on training

1. Fix the graph parameters and train the ADMM layers to see whether it will converge

2. Fix the ADMM layers (as the optimal solutions) and train the Graph Learning Modules

# Work plan and results from Dec 14-19
Graphs:
- [x] use a Dijkstra Algorithm to find the shortest $k$ paths for each node
- [x] extend the $k$-connection to GCN feature extractor (and distance convolution, laplacian embeddings)
- [x] Graph Learning Module with the $k$-connection
- [ ] edit the multiplications in ADMM block
- [ ] test and debug

> Why no concatenate? some leave-out points, 3/4 loops

Training:
- [] Reuse the fixed ADMM with proper parameter

# Works Done by Dec 19
Works done:
1. Edited graph connections: for PEMS04, $k=5$ has better result.
2. Normalized input and outputs: normalized data for training dataset $\tilde{y} = \frac{y-mean}{std}$, normalize output $\hat{x} = ReLU(\tilde{x} \times std + mean)$. Adding ReLU works! No relu leads to failure in training.

3. Train again on PeMS04 (the good):
    - Settings: 5 ADMM blocks, 20 ADMM layers, 5 CGD lters, sigma=6, GNN\_layers = 2
    - learning rate = 0.002
    - trim CGD parameters: (0, 0.2).
    - results: 16 epochs, loss = 1023, RMSE = 43.97, MAE = 19.28. Slightly higher than baselines.

4. Train on PeMS08 (is good for running):
    - use the same settings as PeMS04. Change the learning rate to 0.001.

5. Debug on the PeMS03 dataset.

Results (30 epochs, test dataset, lr=0.001) saved in `/mnt/qij/Dec-Results/logs/4_hop_add_self/PEMS08_MSE_4b25_4h_6f.log` and `/mnt/qij/Dec-Results/logs/5_hop_add_self/PEMS04_MSE_5b25_4h_6f.log`

| Dataset | (nodes, edges) |batch size | ADMM (blocks, layers) | CGD (iters) | k |$\sigma$ | RMSE | MAE | time per epoch|
| :------:|:---:|:--: |:-------------------:| :---------:|:---:|:-----:|:----:|:---:|:---:|
| PeMS04| (307, 680)  |8 |         4, 25        |  3          | 4  | 6     |21.8154 |10.4353 |11min13s |
|PeMS08| (170, 590) |16 | 4, 25 | 3 | 5 | 6 | 19.0935 | 11.6487 | 7min5s|

<!-- Change the activation function in GCN extrapolation to None. PeMS08 trim to (0, 0.18)

| Dataset | (nodes, edges) |batch size | ADMM (blocks, layers) | CGD (iters) | k |$\sigma$ | RMSE | MAE | time per epoch|
| :------:|:---:|:--: |:-------------------:| :---------:|:---:|:-----:|:----:|:---:|:---:|
| PeMS04 (6 epoch) | (307, 680)  |6 |         5, 25        |  3          | 6  | 15     |23.5424 |10.5416 |17min5s |
|PeMS08 (18 epoch)| (170, 590) |12 | 5, 25 | 3 | 4 | 6 | 17.4363 | 7.2175 | 11min13s|

Results in `/mnt/qij/Dec-Results/logs/6_hop_add_self_norelu` -->

More results with $\sigma=20$: has NaN issue in extrapolation (agg doesn't contain NaN, but the result contains NaN.). WHY? it seems that sigma should be quite small so that padding itself doesn't makes sense? Why would the first linear layer have NaN? Gradient explosion?

try with $k=6, \sigma=6$, stable. Results saved in `/mnt/qij/Dec-Results/logs/6_hop_add_self_norelu`.
| Dataset | (nodes, edges) |batch size | ADMM (blocks, layers) | CGD (iters) | k |$\sigma$ | RMSE | MAE | time per epoch|
| :------:|:---:|:--: |:-------------------:| :---------:|:---:|:-----:|:----:|:---:|:---:|
| PeMS04 (12 epoch) | (307, 680)  |6 |         5, 25        |  3          | 6  | 6     |21.7905 |9.2187 |22min44s |
|PeMS08 (12 epoch)| (170, 590) |12 | 5, 25 | 3 | 6 | 6 | 18.6480 | 8.7849 | 11min13s|
|PeMS08 (18 epoch)| (170, 590) |12 | 5, 25 | 3 | 6 | 6 | 17.3584 | 7.2107 | 11min13s|

try with extrapolation with SELU, $\sigma=15$ is ok, $k=6$ is okay. Results in `Dec-Results/logs/6_hop_selu`. So what's the point of the GCN feature extractor if $sigma \ll \min(d)$?

| Dataset | (nodes, edges) |batch size | ADMM (blocks, layers) | CGD (iters) | k |$\sigma$ | RMSE | MAE | time per epoch|
| :------:|:---:|:--: |:-------------------:| :---------:|:---:|:-----:|:----:|:---:|:---:|
| PeMS04 (6 epoch, explode) | (307, 680)  |6 |         5, 25        |  3          | 6  | 15     |23.3915 |11.6454 |23min32s |
|PeMS08 (12 epoch)| (170, 590) |12 | 5, 25 | 3 | 6 | 15 | 20.8812 | 13.1768 | 11min13s|
|PeMS08 (18 epoch)| (170, 590) |12 | 5, 25 | 3 | 6 | 15 | 17.4074 | 7.3773 | 11min13s| 

Try with PeMS04 and $\sigma=3$
| Dataset | (nodes, edges) |batch size | ADMM (blocks, layers) | CGD (iters) | k |$\sigma$ | RMSE | MAE | time per epoch|
| :------:|:---:|:--: |:-------------------:| :---------:|:---:|:-----:|:----:|:---:|:---:|
| PeMS04 (6 epoch, explode) | (307, 680)  |6 |         5, 25        |  3          | 6  | 3     |23.3915 |11.6454 |23min32s |

Is it true that smaller $\sigma$ leads to better result? If so, Feature extraction is not necessary.

Finally, what's the function of $\sigma$?


To discuss:
1. Graph Feature Extractor: for now, we use $w_{ij}=-\exp(-\frac{d^2(i,j)}{\sigma^2})$, but different graphs has different $d$:
    - in PeMS04 and PeMS08, $d\sim 10^0 - 10^2$
    - in PeMS03 and PeMS07, $d \sim 10^{-1} - 10^1$, actually costs but not 
    
    For PeMS04 we set $\sigma=6$ and the weights actually falls to close to 0 or 1 (also difficult to normalize). Only when $d < 18$ the weights would be larger than 1e-4. **Larger d will cause NaN in graph learning module. BUT WHY?** So we want to use other feature convolutions:
    
    - sum on every neighbors: use the Adjacency graph for convolution. (if adjustable, use mask matrix?)
    - mean on every neighbors?
    - concatenate? Problem: padding zeros. From smallest to largest?
    - No GCN module, use only Spatial embeddings (graph embedding methods? or learnable embeddings.)


2. CGD parameters: when trimmed to (0, 0.2). Nearly all parameters are trimmed. Why?
    - Problem 1: For PeMS03 dataset, the first few epochs will get NaN in $Qf$. *GUESS*: a large input feature $f$ occurs in the second DGL/UGL.

3. Why PeMS04 is good for training?
    - $\rho$ s, $Q, M$ are stable.
    - trim $(\alpha,\beta)$ s to (0, 0.2), and nearly all alphas are trimmed. and do not move at all. *Actually makes some of the gradients not moving and some of the gradients move with fixed momentum gradient descent?*
    - carefully select the value for sigma.

Work plan:
1. For PeMS04 and PeMS08, try with different parameters?
2. For PeMS03 and PeMS07, try with different $\sigma$.
3. Try to find out why full connection doesn't work.

# Discussion on Dec 19

# Discussion on Jan 3
Problem:
(1) First extrapolation always gets NaN value in `self.shrink`

Possible reasons?
- gradient explosion?
- current problem is in PEMS07.
Counter output: Counter({2: 800, 1: 59, 3: 23, 4: 1})

Results:

`python train.py --cuda 1 --dataset PEMS03 --batchsize 8 --hop 4 --tin 12 --tout 12` saved in `Dec-Results/logs/4_hop_concatFE_12_12/PEMS03_MSE_5b25_4h_6f.log`, in tmux `pems03-12-12`

`python train.py --cuda 0 --dataset PEMS07 --batchsize 6` saved in `Dec-Results/logs/6_hop_concatFE_6_6/PEMS07_MSE_5b25_4h_6f.log` and in tmux `pems04-12-12` (Failed)

change into `python train.py --cuda 0 --dataset PEMS07 --batchsize 8 --hop 3`, change Graph Aggregation Layer with SELU

Current modification: changed linear extrapolation with Graph Aggregation Layer

先把层数减少点再说。爆炸原因：Ldr T x出现NaN。为什么？如何debug？

对PEMS08 的 12-12 训练：assert not torch.isnan(agg).any(), 'extrapolation agg has nan value'

Graph aggregation layer: