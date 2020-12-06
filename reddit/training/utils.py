import random
import numpy as np
import torch
import dgl
from random import sample

def set_seed(args=None):
    seed = 1 if not args else args.seed
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)
    
    
def sample_node(num_sample, all_nodes):
    
    list_nodes = [i for i in range(all_nodes)]
    list_sample = sample(list_nodes, num_sample)

    return list_sample

