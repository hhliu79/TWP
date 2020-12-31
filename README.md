# PyTorch implementation of TWP
[Overcoming Catastrophic Forgetting in Graph Neural Networks](https://arxiv.org/pdf/2012.06002.pdf), AAAI2021

# 
![image](https://github.com/hhliu79/TWP/blob/master/overview.png)

# Cite
If you find this code useful in your research, please consider citing:

    @inproceedings{liu2021overcoming,
	Title = {Overcoming Catastrophic Forgetting in Graph Neural Networks},
	Author = {Huihui Liu, Yiding Yang, and Xinchao Wang},
	Booktitle  = {AAAI Conference on Artificial Intelligence},
	Year = {2021}
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

# Results
| Method | AP | AF |
| :---:         |     :---:      |     :---:    |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |
