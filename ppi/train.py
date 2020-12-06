import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from dgl.data import register_data_args
from models.model_factory import get_model
from models.utils import evaluate
from training.utils import set_seed
from dataset.load_ppi import load_ppi_dataset, continuum_ppi_dataset
from dataset.utils import semi_task_manager
from dataset.nei_sam import NeighborSampler
import importlib
import dgl
import os
from torch.utils.data import DataLoader
import copy

def main(args):
  
    cur_step = 0
    best_score = -1
    best_loss = 10000
    loss_fcn = torch.nn.BCEWithLogitsLoss()    
    torch.cuda.set_device(args.gpu)   
    
    dataset = load_ppi_dataset(args)
    train_g, train_dataloader, valid_dataloader, test_dataloader, n_classes, num_feats, num_train, num_valid, num_test = dataset  
    continuum_data = continuum_ppi_dataset(dataset, args)
    task_manager = semi_task_manager(continuum_data.dataset_info())
    
    model = get_model(args, dataset, model_name="GAT").cuda()
    life_model = importlib.import_module(f'LifeModel.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)
    life_model_ins.dataloader = train_dataloader

    score_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    for current_task in range(args.n_tasks):
        #task_manager.add_task(current_task, {"test_mask": None})
        offset1, offset2 = task_manager.get_label_offset(current_task) 
        for epoch in range(args.epochs):  
            loss_list = []
            for batch, data in enumerate(train_dataloader):
                subgraph, feats, labels = data
                
                lbl = labels[:, offset1:offset2]
                lbl_sum = torch.sum(lbl, 1)                             
                mask = torch.BoolTensor( [l >= 0 for l in lbl_sum] )
                num = torch.sum((mask == True),0)
                if num == 0:
                    continue
                feats = feats.cuda()
                labels = labels.cuda()

                if args.method == 'lwf':
                    loss = life_model_ins.observe(subgraph, feats, labels, current_task, mask, prev_model)
                elif args.method == 'gem':
                    loss = life_model_ins.observe(subgraph, feats, labels, current_task, mask, epoch)
                else:
                    loss = life_model_ins.observe(subgraph, feats, labels, current_task, mask)
                loss_list.append(loss.item())
                
            loss_data = np.array(loss_list).mean()
            print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))
             
            if epoch % 5 == 0:
                score_list = []
                val_loss_list = []
                for batch, valid_data in enumerate(valid_dataloader):
                    subgraph_val, feats_val, labels_val = valid_data
                    
                    lbl = labels_val[:, offset1:offset2]
                    lbl_sum = torch.sum(lbl, 1)                             
                    mask_val = torch.BoolTensor( [l >= 0 for l in lbl_sum] )
                    num = torch.sum((mask_val == True),0)
                    if num == 0:
                        continue
                
                    feats_val = feats_val.cuda()
                    labels_val = labels_val.cuda()
                    labels_val = labels_val[mask_val, offset1: offset2]
                    score, val_loss = evaluate(feats_val, model, subgraph_val, labels_val, loss_fcn, mask_val, offset1, offset2)

                    score_list.append(score)
                    val_loss_list.append(val_loss)
                mean_score = np.array(score_list).mean()
                mean_val_loss = np.array(val_loss_list).mean()
                print("Val F1-Score: {:.4f} ".format(mean_score))
                
                # early stop
                if mean_score > best_score or best_loss > mean_val_loss:
                    path = 'weights/' + args.method + '.pkl'
                    torch.save(model.state_dict(), path)
                    best_score = np.max((mean_score, best_score))
                    best_loss = np.min((best_loss, mean_val_loss))
                    cur_step = 0
                else:
                    cur_step += 1
                    if cur_step == args.patience:
                        print(epoch)
                        model.load_state_dict(torch.load(path))        
                        break        
                        
        if args.method == 'ewc' or args.method == 'mas' or args.method == 'twp':       
            life_model_ins.compute_gradient(subgraph, feats, labels, current_task, mask)
        
        score_mean = []
        for task_i in range(current_task+1):
            offset1_test, offset2_test = task_manager.get_label_offset(task_i)

            test_score_list = []
            for batch, test_data in enumerate(test_dataloader):
                subgraph, feats, labels = test_data
                
                lbl = labels[:, offset1_test : offset2_test]
                lbl_sum = torch.sum(lbl, 1)                             
                mask = torch.BoolTensor( [l >= 0 for l in lbl_sum] )
                num = torch.sum((mask == True),0)
                if num < 10:
                    continue
                
                feats = feats.cuda()
                labels = labels.cuda()
                labels = labels[mask, offset1_test: offset2_test]                

                score, _ = evaluate(feats.float(), model, subgraph, labels, loss_fcn, mask, offset1_test, offset2_test)
                test_score_list.append(score) 
            test_score = np.array(test_score_list).mean()

            print(f"T{task_i:02d}:{test_score*100:.1f}|", end="")
            score_mean.append(test_score)
            score_matrix[current_task][task_i] = test_score

        accs = score_mean[:task_i+1]
        meana = round(np.mean(accs)*100,2)
        meanas.append(meana)
        mean_score = round(np.mean(score_mean)*100,1)
        print(f"mean_score: {mean_score}", end="")
        print()
        prev_model = copy.deepcopy(life_model_ins).cuda()
    
    print('AP: ', mean_score)

    backward = []
    for t in range(args.n_tasks-1):
        b = score_matrix[args.n_tasks-1][t]-score_matrix[t][t]
        backward.append(round(b*100, 2))
    mean_backward = round(np.mean(backward),2)        
    print('AF: ', mean_backward)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=4,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for exp")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--n-tasks', type=int, default=12,
                        help="number of tasks")
    parser.add_argument('--file', type=str, default='ppi.txt',
                        help=" result ")
    parser.add_argument('--patience', type=int, default=10,
                        help="used for early stop")
    parser.add_argument('--basemodel', type=str, default='GAT',
                        help="basemodel")
    parser.add_argument('--method', type=str, choices=["finetune",'lwf', 'gem', 'ewc', 'mas', 'twp', 'jointtrain'], default="twp",
                        help="which lifelong method is adopted, 'twp' is our method")


    # parameters for GAT model
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--batch-size', type=int, default=2,
                        help = "1000 for reddit, 2 for PPI")
        
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
    
   
    args = parser.parse_args()
    print(args)
    set_seed(args)
    main(args)