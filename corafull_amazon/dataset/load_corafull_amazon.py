import torch
import dgl
from dgl.data import register_data_args, load_data, CoraFull, AmazonCoBuy
from dgl import DGLGraph
import networkx as nx
import collections
import numpy as np
from torch.autograd import Variable

def load_corafull():
    # load corafull dataset
    CoraFull_dataset = CoraFull()
    g = next(iter(CoraFull_dataset))
    label = g.ndata['label'].numpy()
    label_counter = collections.Counter(label)
    selected_ids = [id for id, count in label_counter.items() if count > 150]
    np.random.shuffle(selected_ids)
    print(g)
    print(f"selected {len(selected_ids)} ids from {max(label)+1}")
    
    mask_map = np.array([label == id for id in selected_ids])
    # set label to -1 and remap the selected id
    label = label * 0.0 - 1
    label = label.astype(np.int)
    for newid, remap_map in enumerate(mask_map):
        label[remap_map] = newid
        g.ndata['label'] = torch.LongTensor(label)
    mask_map = np.sum(mask_map, axis=0)
    mask_map = (mask_map >= 1).astype(np.int)

    mask_index = np.where(mask_map == 1)[0]
    np.random.shuffle(mask_index)

    train_mask = np.zeros_like(label)
    train_mask[ mask_index[ 0 : 40*len(selected_ids) ] ] = 1
    val_mask = np.zeros_like(label)
    val_mask[ mask_index[ 40*len(selected_ids) : 60*len(selected_ids) ] ] = 1
    test_mask = np.zeros_like(label)
    test_mask[ mask_index[ 60*len(selected_ids): ] ] = 1
    
    train_mask = torch.BoolTensor(train_mask).cuda()
    val_mask = torch.BoolTensor(val_mask).cuda()
    test_mask = torch.BoolTensor(test_mask).cuda()
    
    labels = g.ndata['label'].cuda()
    features = g.ndata['feat'].cuda()
    
    return g, features, labels, train_mask, val_mask, test_mask


def load_AmazonCoBuy(name):
    # load corafull dataset
    dataset = AmazonCoBuy('computers')
    g = next(iter(dataset))
    label = g.ndata['label'].numpy()
    label_counter = collections.Counter(label)
    selected_ids = [id for id, count in label_counter.items()]
    np.random.shuffle(selected_ids)
    print(f"selected {len(selected_ids)} ids from {max(label)+1}")
    
    mask_map = np.array([label == id for id in selected_ids])
    # set label to -1 and remap the selected id
    label = label * 0.0 - 1
    label = label.astype(np.int)
    for newid, remap_map in enumerate(mask_map):
        label[remap_map] = newid
        g.ndata['label'] = torch.LongTensor(label)
    mask_map = np.sum(mask_map, axis=0)
    mask_map = (mask_map >= 1).astype(np.int)

    mask_index = np.where(mask_map == 1)[0]
    np.random.shuffle(mask_index)

    train_mask = np.zeros_like(label)
    train_mask[ mask_index[ 0 : 40*len(selected_ids) ] ] = 1
    val_mask = np.zeros_like(label)
    val_mask[ mask_index[ 40*len(selected_ids) : 60*len(selected_ids) ] ] = 1
    test_mask = np.zeros_like(label)
    test_mask[ mask_index[ 60*len(selected_ids): ] ] = 1
    
    train_mask = torch.BoolTensor(train_mask).cuda()
    val_mask = torch.BoolTensor(val_mask).cuda()
    test_mask = torch.BoolTensor(test_mask).cuda()

    labels = g.ndata['label'].cuda()
    features = g.ndata['feat'].cuda()
    
    return g, features, labels, train_mask, val_mask, test_mask
    

def load_corafull_amazon_dataset(args):

    if args.dataset.lower() == 'corafull':
        g, features, labels, train_mask, val_mask, test_mask = \
                                        load_corafull()
    elif args.dataset.lower() == 'amazoncobuy':
        g, features, labels, train_mask, val_mask, test_mask = \
                                        load_AmazonCoBuy(args.dataset.lower())
    
    print(f"""--------Data statistics--------
      #Edges {g.number_of_edges()}
      #Features {features.shape[1]}
      #Classes {max(labels) + 1} 
      #Train samples {train_mask.int().sum().item()}
      #Val samples {val_mask.int().sum().item()}
      #Test samples {test_mask.int().sum().item()}""")
    
    return (g, features, labels, train_mask, val_mask, test_mask)


class continuum_corafull_amazon_dataset:
    """
    convert the semi-supervised dataset to "n_task" continuum dataset
    """
    def __init__(self, dataset, args):
        self.n_tasks = args.n_tasks
        self.current_task = 0
        self.dataset = dataset

        g, features, labels, train_mask, val_mask, test_mask \
                                                    = self.dataset
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

    def __next__(self):
        if self.current_task >= self.n_tasks:
            raise StopIteration
        else:
            current_task = self.current_task
            labels_of_current_task = self.labels_of_tasks[current_task]
            g, features, labels, train_mask, val_mask, test_mask \
                                                    = self.dataset
           
            conditions = torch.BoolTensor( [l in labels_of_current_task for l in labels.detach().cpu()] )
            mask_of_task = torch.where(conditions, 
                                        torch.tensor(1), 
                                        torch.tensor(0) )
            mask_of_task = mask_of_task.cuda()
            
            train_mask = (train_mask.to(torch.long) * mask_of_task).to(torch.bool)
            val_mask = (val_mask.to(torch.long) * mask_of_task).to(torch.bool)
            test_mask = (test_mask.to(torch.long) * mask_of_task).to(torch.bool)
            
            self.current_task += 1
            return current_task, (g, features, labels, train_mask, val_mask, test_mask)
            

if __name__ == "__main__":
    pass