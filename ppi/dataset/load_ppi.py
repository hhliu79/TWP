import torch
import dgl
from dgl.data import register_data_args, load_data, LegacyPPIDataset
from torch.utils.data import DataLoader
from dgl import DGLGraph
import networkx as nx
import collections
import numpy as np

def collate(sample):
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

def load_ppi(args):

    train_dataset = LegacyPPIDataset(mode='train')
    valid_dataset = LegacyPPIDataset(mode='valid')
    test_dataset = LegacyPPIDataset(mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate)
    
    n_classes = train_dataset.labels.shape[1]
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph

    num_train = train_dataset.features.shape[0]
    num_valid = valid_dataset.features.shape[0]
    num_test = test_dataset.features.shape[0]

    return g, train_dataloader, valid_dataloader, test_dataloader, n_classes, num_feats, num_train, num_valid, num_test
    #return g, features, labels, train_mask, val_mask, test_mask


def load_ppi_dataset(args):
    
    g, train_dataloader, valid_dataloader, test_dataloader, n_classes, num_feats, num_train, num_valid, num_test = load_ppi(args)
    
    print(f"""--------Data statistics--------
      #Edges {g.number_of_edges()}
      #Features {num_feats}
      #Classes {n_classes} 
      #Train samples {num_train}
      #Val samples {num_valid}
      #Test samples {num_test}""")
    
    return (g, train_dataloader, valid_dataloader, test_dataloader, n_classes, num_feats, num_train, num_valid, num_test)


class continuum_ppi_dataset:
    """
    convert the semi-supervised dataset to "n_task" continuum dataset
    """
    def __init__(self, dataset, args):
        self.n_tasks = args.n_tasks
        self.current_task = 0
        self.dataset = dataset
        
        train_g, train_dataloader, valid_dataloader, test_dataloader, n_classes, num_feats, num_train, num_valid, num_test = dataset  

        n_labels = n_classes
        self.n_labels_per_task = n_labels // self.n_tasks
        self.labels_of_tasks = {}
        for task_i in range(self.n_tasks):
            self.labels_of_tasks[task_i] = list(range(task_i * self.n_labels_per_task, (task_i+1) * self.n_labels_per_task))
    
    def dataset_info(self):
        return {'n_labels_per_task': self.n_labels_per_task, 
                'n_tasks': self.n_tasks}

if __name__ == "__main__":
    pass