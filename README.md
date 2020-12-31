# PyTorch implementation of TWP
Overcoming Catastrophic Forgetting in Graph Neural Networks, AAAI2021

# 
![image](https://github.com/hhliu79/TWP/blob/master/overview.png)

# Cite
@article{liu2020overcoming,
  title={Overcoming Catastrophic Forgetting in Graph Neural Networks},
  author={Liu, Huihui and Yang, Yiding and Wang, Xinchao},
  year={2021},
  booktitle={AAAI Conference on Artificial Intelligence},
}

# Dependencies
See the file [requirements.txt](https://github.com/hhliu79/TWP/blob/master/requirements.txt) for more information about how to install the dependencies.

# Datasets
## Node classification
We conduct experiments on four public datasets (Corafull, Amazon Computer, PPI, Reddit) based on [DGL](https://docs.dgl.ai/en/0.4.x/).<br>

## Graph classification
We conduct experiments on one public dataset (Tox21) based on [DGLlife](https://lifesci.dgl.ai/index.html).

# Models
We use [DGL](https://docs.dgl.ai/en/0.4.x/) to implement all the GNN models.

# Overview
Here we provide the implementation of our method based on the Corafull dataset. The folder `< corafull_amazon/ >` is organised as follows:
* `< LifeModel/ >` contains the implementation of all the continual learning method for GNNs, including the baseline methods and our method;
* `< dataset/ >`  contains code to load the dataset; 
* `< models/ >` contains the implementation of the GNN backbone;
* `< training/ >` contains code to set seed;
* The file `< train.py >` is used to execute the whole training process on the Corafull dataset;
* The file `< run.sh >` contains an example to run the code.
