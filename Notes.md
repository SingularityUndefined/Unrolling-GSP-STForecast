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

Results (30 epochs, test dataset, lr=0.001)

| Dataset | (nodes, edges) |batch size | ADMM (blocks, layers) | CGD (iters) | k |sigma | RMSE | MAE | time per epoch|
| :------:|:---:|:--: |:-------------------:| :---------:|:---:|:-----:|:----:|:---:|:---:|
| PeMS04| (307, 680)  |8 |         5, 25        |  3          | 5  | 6     |21.8154 |10.4353 |17min5s |
|PeMS08| (170, 590) |16 | 5, 25 | 3 | 4 | 6 | 19.0935 | 11.6487 | 7min5s|


To discuss:
1. Graph Feature Extractor: for now, we use $w_{ij}=-\exp(-\frac{d^2(i,j)}{\sigma^2})$, but different graphs has different $d$:
    - in PeMS04 and PeMS08, $d\sim 10^0 - 10^2$
    - in PeMS03 and PeMS07, $d \sim 10^{-1} - 10^1$
    
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