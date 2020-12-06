import argparse
import numpy as np
import torch
from dgl.data import register_data_args
import dgl
from models.model_factory import get_model
from models.utils import evaluate
from training.utils import set_seed
from dataset.load_reddit import load_reddit_dataset, continuum_reddit_dataset, load_subtensor
from dataset.utils import semi_task_manager
from dataset.nei_sam import NeighborSampler
import importlib
import os
from torch.utils.data import DataLoader
import copy

def main(args):
    
    torch.cuda.set_device(args.gpu)   
    
    dataset = load_reddit_dataset(args)
    g, features, labels, train_mask, val_mask, test_mask, train_nid = dataset  

    datas = [labels, train_mask, val_mask, test_mask, train_nid]
    continuum_data = continuum_reddit_dataset(datas, args)
    task_manager = semi_task_manager(continuum_data.dataset_info())
    
    
    model = get_model(args, model_name="GAT").cuda()
    life_model = importlib.import_module(f'LifeModel.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)
    
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    for task_i, dataset_i in enumerate(continuum_data):
        
        current_task, (train_mask, val_mask, test_mask, train_task_nid) = dataset_i
        task_manager.add_task(current_task, {"test_mask": test_mask})
        task_manager.add_g(g)
        label_offset1, label_offset2 = task_manager.get_label_offset(current_task) # 0 5   
        
        dataloader = DataLoader(
            dataset=train_task_nid.cpu().numpy(),
            batch_size=args.batch_size,
            collate_fn=sampler.sample_blocks,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)  

        ############### update memory for gem
        if args.method == 'gem':
            train_nid = train_task_nid[:args.n_memories]
            loader = DataLoader(
                    dataset=train_nid.cpu().numpy(),
                    batch_size=train_nid.shape[0],
                    collate_fn=sampler.sample_blocks,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.num_workers)

            for step, blocks_ in enumerate(loader):
                input_nodes = blocks_[0].srcdata[dgl.NID]   
                seeds = blocks_[-1].dstdata[dgl.NID]
                batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes) 

                life_model_ins.memory_data['block'].append(blocks_)
                life_model_ins.memory_data['feature'].append(batch_inputs)
                life_model_ins.memory_data['label'].append(batch_labels)

        ################
    
        labels_of_current_task = task_manager.labels_of_tasks[task_i]  
        for epoch in range(args.epochs):  
            loss_list = []
            for step, blocks in enumerate(dataloader):
                input_nodes = blocks[0].srcdata[dgl.NID]   
                seeds = blocks[-1].dstdata[dgl.NID]
                batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes)
                
                mask = torch.BoolTensor( [l in labels_of_current_task for l in batch_labels.detach().cpu()] )
                num = torch.sum((mask == True),0)
                if num < 10:
                    continue
                
                if args.method == 'lwf':
                    loss = life_model_ins.observe(blocks, batch_inputs, batch_labels, task_i, prev_model)
                else:
                    loss = life_model_ins.observe(blocks, batch_inputs, batch_labels, task_i)
                loss_list.append(loss.item())
                
            loss_data = np.array(loss_list).mean()
            print("Epoch {:04d} | Loss: {:.4f}".format(epoch + 1, loss_data))
                

        if args.method == 'ewc' or args.method == 'mas' or args.method == 'twp':      
            life_model_ins.compute_gradient(blocks, batch_inputs, batch_labels, task_i)
        
        acc_mean = []
        for t in range(task_i+1):
            test_mask = task_manager.retrieve_task(t)['test_mask']
            label_offset1, label_offset2 = task_manager.get_label_offset(t)

            acc = evaluate(model, g, features, labels, test_mask, args.batch_size, label_offset1, label_offset2)
            acc_matrix[task_i][t] = round(acc*100,2)
            acc_mean.append(acc)
            print(f"T{t:02d}:{acc*100:.1f}|", end="")
        
        accs = acc_mean[:task_i+1]
        meana = round(np.mean(accs)*100,2)
        meanas.append(meana)
        
        acc_mean = round(np.mean(acc_mean)*100,1)
        print(f"acc_mean: {acc_mean}", end="")
        print()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    print('AF: ', mean_backward)

        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=5,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for exp")
    parser.add_argument("--epochs", type=int, default=30,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--n-tasks', type=int, default=8,
                        help="number of tasks")
    parser.add_argument('--basemodel', type=str, default='GAT',
                        help="basemodel")
    parser.add_argument('--method', type=str, choices=["finetune",'lwf', 'gem', 'ewc', 'mas', 'twp', 'jointtrain'], default="twp",
                        help="which lifelong method is adopted, 'twp' is our method")


    # parameters for GAT model
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")

    
    # parameters for ewc/mas/gem
    parser.add_argument('--memory-strength', type=float, default=10000,
                        help="memory strength, 10000 for ewc/mas/twp, 0.5 for gem")
    parser.add_argument('--n-memories', type=int, default=1000,
                        help="number of memories, for gem")
    
    # parameters for our method (twp)
    parser.add_argument('--lambda_l', type=float, default=10000, 
                        help=" ")    
    parser.add_argument('--lambda_t', type=float, default=10000, 
                        help=" ")    
    parser.add_argument('--beta', type=float, default=0.01, 
                        help=" ")

    # for reddit dataset
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                            'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                   help='Normalization method for the adjacency matrix.')
    parser.add_argument('--sparse', type=bool, default=True,
                        help='')
    parser.add_argument('--fan-out', type=str, default='10,25')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")

    args = parser.parse_args()
    print(args)
    set_seed(args)
    main(args)