## AIDE:Attack Inference Based on HeterogeneousDependency Graphs

This code is the pytorch implementation of our preprinted paper,  AIDE:Attack Inference Based on HeterogeneousDependency Graphs with MITRE ATT&CK.

## Installation

This code runs on pytorch 2.1, python 3.9 and dgl 2.2. 


## Environment

The implementation is supposed to train in the GPU enviornment. 
I test all of the datasets with RGCN on GeForce RTX 4060 and CPU with 32GB RAM.


## Usage

It requires original () data, which can be downloaded [here][data_seg] for segmentation.  We also provide our [processed one][data_pre] but we don't guarantee its compatibility.

The train.py is quite easy to read, I'm sure you can run and test it smoothly.

## Note

We test our code on pycharm at first, so I suspect some part could be missing. If something went wrong, please contact me by email.


[cng]: https://github.com/mdeff/cnn_graph
[arxiv]: https://arxiv.org/abs/1806.02952
[data_seg]: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
[data_pre]: https://1drv.ms/f/s!Am_uh1epJzCIjQeZviRjHa4fCkFy
