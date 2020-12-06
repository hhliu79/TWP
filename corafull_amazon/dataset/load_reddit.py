import torch
import dgl
from dgl.data import register_data_args, load_data, RedditDataset
from dgl import DGLGraph
import networkx as nx
import collections
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])

def load_reddit(args):
    
    data = RedditDataset(self_loop = False)
    train_mask = data.train_mask
    test_mask = data.test_mask
    val_mask = data.val_mask
    
    features = torch.Tensor(data.features)
    in_feats = features.shape[1]
    labels = torch.LongTensor(data.labels)
    n_classes = data.num_labels
    
    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    g.ndata['features'] = features
    prepare_mp(g)

    train_nid = torch.LongTensor(np.nonzero(train_mask)[0])
    val_nid = torch.LongTensor(np.nonzero(val_mask)[0])
    train_mask = torch.BoolTensor(train_mask).cuda()
    test_mask = torch.BoolTensor(test_mask).cuda()
    val_mask = torch.BoolTensor(val_mask).cuda()
   
    return g, features, labels, train_mask, val_mask, test_mask, train_nid
    
    
def load_subtensor(g, labels, seeds, input_nodes):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].cuda()
    batch_labels = labels[seeds].cuda()
    return batch_inputs, batch_labels  

    
def load_reddit_dataset(args):
    
    g, features, labels, train_mask, val_mask, test_mask, train_nid = \
                                        load_reddit(args)
    
    print(f"""--------Data statistics--------
      #Edges {g.number_of_edges()}
      #Features {features.shape[1]}
      #Classes {max(labels) + 1} 
      #Train samples {train_mask.int().sum().item()}
      #Val samples {val_mask.int().sum().item()}
      #Test samples {test_mask.int().sum().item()}""")
    
    return (g, features, labels, train_mask, val_mask, test_mask, train_nid)


class continuum_reddit_dataset:
    """
    convert the semi-supervised dataset to "n_task" continuum dataset
    """
    def __init__(self, dataset, args):
        self.n_tasks = args.n_tasks
        self.current_task = 0
        self.dataset = dataset
        self.label_list = []
        labels, train_mask, val_mask, test_mask, train_nid = dataset
        
        n_labels = max(labels) + 1
        self.n_labels_per_task = n_labels // self.n_tasks
        self.labels_of_tasks = {}
        for task_i in range(self.n_tasks):
            self.labels_of_tasks[task_i] = list(range( task_i * self.n_labels_per_task, (task_i+1) * self.n_labels_per_task ))
    
    def dataset_info(self):
        return {'n_labels_per_task': self.n_labels_per_task, 
                'n_tasks': self.n_tasks,
                'labels_of_tasks': self.labels_of_tasks}

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()
    
    # for gem and combined
    def __next__(self):
        if self.current_task >= self.n_tasks:
            raise StopIteration
        else:
            current_task = self.current_task
            labels_of_current_task = self.labels_of_tasks[current_task]
            labels, train_mask, val_mask, test_mask, train_nid = self.dataset
            
            self.label_list = self.label_list + labels_of_current_task
            conditions_train = torch.BoolTensor( [l in self.label_list for l in labels.detach().cpu()] )
            mask_of_task_train = torch.where(conditions_train, 
                                        torch.tensor(1), 
                                        torch.tensor(0) )
            mask_of_task_train = mask_of_task_train.cuda()
            train_mask = (train_mask.to(torch.long) * mask_of_task_train).to(torch.bool)            
            train_task_nid = torch.cuda.LongTensor(np.nonzero(train_mask))
            train_task_nid = torch.squeeze(train_task_nid, -1)
            
            
            conditions = torch.BoolTensor( [l in labels_of_current_task for l in labels.detach().cpu()] )
            mask_of_task = torch.where(conditions, 
                                        torch.tensor(1), 
                                        torch.tensor(0) )
            mask_of_task = mask_of_task.cuda()
            val_mask = (val_mask.to(torch.long) * mask_of_task).to(torch.bool)
            test_mask = (test_mask.to(torch.long) * mask_of_task).to(torch.bool)
            
            self.current_task += 1
            return current_task, (train_mask, val_mask, test_mask, train_task_nid)
    
if __name__ == "__main__":
    pass