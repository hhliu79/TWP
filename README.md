# PyTorch implementation of TWP
Overcoming Catastrophic Forgetting in Graph Neural Networks, AAAI2021

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
Here we provide the implementation of our method based on the Corafull dataset. The repository is organised as follows:
* `< LifeModel/ >`  
* dataset/ contains the necessary dataset files for Cora
* models/
* training/
