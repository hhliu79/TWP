import argparse
import numpy as np
import networkx as nx
import torch
from dgl.data import register_data_args
from models.model_factory import get_model
from models.utils import evaluate
from training.utils import set_seed
from dataset.load_corafull_amazon import load_corafull_amazon_dataset, continuum_corafull_amazon_dataset
from dataset.utils import semi_task_manager 
import importlib
import dgl
import copy

def main(args):

    torch.cuda.set_device(args.gpu)
    dataset = load_corafull_amazon_dataset(args)
    continuum_data = continuum_corafull_amazon_dataset(dataset, args)

    task_manager = semi_task_manager(continuum_data.dataset_info())
    g, features, labels, train_mask, val_mask, test_mask = dataset
    task_manager.add_g(g)
    
    model = get_model(dataset, args, task_manager).cuda()
    life_model = importlib.import_module(f'LifeModel.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)
    
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    for task_i, dataset_i in enumerate(continuum_data):
        current_task, (g, features, labels, train_mask, val_mask, test_mask) = dataset_i
        task_manager.add_task(current_task, {"test_mask": test_mask})
        label_offset1, label_offset2 = task_manager.get_label_offset(current_task)
        
        dur = []
        for epoch in range(args.epochs):
            if args.method == 'lwf':
                life_model_ins.observe(features, labels, task_i, train_mask, prev_model)
            else:
                life_model_ins.observe(features, labels, task_i, train_mask)

        acc_mean = []
        for t in range(task_i+1):
            test_mask = task_manager.retrieve_task(t)['test_mask']
            label_offset1, label_offset2 = task_manager.get_label_offset(t)

            acc = evaluate(model, features, labels, test_mask, label_offset1, label_offset2)
            acc_matrix[task_i][t] = round(acc*100,2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc*100:.2f}|", end="")
            
        accs = acc_mean[:task_i+1]
        meana = round(np.mean(accs)*100,2)
        meanas.append(meana)
        
        acc_mean = round(np.mean(acc_mean)*100,2)
        print(f"acc_mean: {acc_mean}", end="")
        print()       
        prev_model = copy.deepcopy(life_model_ins).cuda()
    
    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)        
    print('AF: ', mean_backward)

            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for exp")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs, default = 200")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--n-tasks', type=int, default=9,
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
    parser.add_argument('--patience', type=int, default=30,  
                        help='')
    
    # parameters for ewc/mas/gem
    parser.add_argument('--memory-strength', type=float, default=10000,
                        help="memory strength, 10000 for ewc/mas/twp, 0.5 for gem")
    parser.add_argument('--n-memories', type=int, default=100,
                        help="number of memories, for gem")
    
    # parameters for our method (twp)
    parser.add_argument('--lambda_l', type=float, default=10000, 
                        help=" ")    
    parser.add_argument('--lambda_t', type=float, default=10000, 
                        help=" ")    
    parser.add_argument('--beta', type=float, default=0.01, 
                        help=" ")

    args = parser.parse_args()
    set_seed(args)
    main(args)