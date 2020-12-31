# PyTorch implementation of TWP
[Overcoming Catastrophic Forgetting in Graph Neural Networks](https://arxiv.org/pdf/2012.06002.pdf), AAAI2021

# Method Overview
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
Here we shown the performance comparison on different datasets with GATs as the backbone. For the task performance, we use classification accuracy on Corafull and Amazon Computers datasets, and micro-averaged F1 score for PPI and Reddit datasets. The symbol↑(↓) indicates higher (lower) is better. 

## Dataset: Corafull

| Method | AP (↑) | AF(↓) |
| :---:         |     :---:      |     :---:    |
| Fine-tune   | 51.6±6.4%     | 46.1±7.0%    |
| LWF     | 57.3±2.3%     | 39.5±3.1%       |
| GEM   | 84.4±1.1%     | 4.2±1.0%    |
| EWC     | 86.9±1.7%     | 6.4±1.8%       |
| MAS   | 84.1±1.8%     | 8.6±2.2%    |
| Ours   | 89.0±0.8%     | 3.3±0.3%       |
| Joint train   | 91.9±0.8%     | 0.1±0.2%       |

## Dataset: Amazon Computers
| Method | AP (↑) | AF(↓) |
| :---:         |     :---:      |     :---:    |
| Fine-tune   | 86.5±8.0%     | 12.3±12.3%    |
| LWF     | 90.3±6.4%     | 9.9±7.0%       |
| GEM   | 97.1±0.9%     | 0.7±0.5%    |
| EWC     | 94.5±3.3%     | 4.6±4.5%       |
| MAS   | 94.0±5.5%     | 5.0±6.9%    |
| Ours   | 97.3±0.6%     | 0.6±0.2%       |
| Joint train   | 98.2±0.6%     | 0.02±0.1%       |

## Dataset: PPI
| Method | AP (↑) | AF(↓) |
| :---:         |     :---:      |     :---:    |
| Fine-tune   | 0.365±0.024%     | 0.178±0.019%    |
| LWF     | 0.382±0.024%     | 0.185±0.060%     |
| GEM   | 0.741±0.016%     | 0.112±0.030%   |
| EWC     | 0.826±0.027%     | 0.142±0.028%     |
| MAS   | 0.749±0.007%     | 0.092±0.008%   |
| Ours   | 0.853±0.004%     | 0.086±0.005%     |
| Joint train   | 0.931±0.40%     | 0.035±0.026%       |

## Dataset: Reddit
| Method | AP (↑) | AF(↓) |
| :---:         |     :---:      |     :---:    |
| Fine-tune   | 0.474±0.006%     | 0.580±0.007%    |
| LWF     | 0.500±0.033%     | 0.550±0.034%     |
| GEM   | 0.947±0.001%     | 0.030±0.008%   |
| EWC     | 0.944±0.019%     | 0.032±0.021%     |
| MAS   | 0.865±0.031%     | 0.085±0.024%   |
| Ours   | 0.954±0.014%     | 0.014±0.015%     |
| Joint train   | 0.978±0.001%     | 0.001±0.001%       |
