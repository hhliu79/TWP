import numpy as np
import torch
import importlib
from dgllife.model import load_pretrained
from dgllife.utils import EarlyStopping, Meter
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import copy
from utils import set_random_seed, load_dataset_for_classification, collate_molgraphs, load_model, load_mymodel

def predict(args, model, bg):
    node_feats = bg.ndata.pop(args['node_data_field']).cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).cuda()
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer, task_i):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.cuda(), masks.cuda()
        logits = predict(args, model, bg)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Mask non-existing labels
        loss = loss_criterion(logits, labels) * (masks != 0).float()
        loss = loss[:,task_i].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
    
    train_score = np.mean(train_meter.compute_metric(args['metric_name']))
    

def run_an_eval_epoch(args, model, data_loader, task_i):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.cuda()
            logits = predict(args, model, bg)
            if isinstance(logits, tuple):
                logits = logits[0]
            eval_meter.update(logits, labels, masks)

    return eval_meter.compute_metric(args['metric_name'])[task_i]


def run_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.cuda()
            logits = predict(args, model, bg)
            if isinstance(logits, tuple):
                logits = logits[0]
            eval_meter.update(logits, labels, masks)

        test_score =  eval_meter.compute_metric(args['metric_name'])
        score_mean = round(np.mean(test_score),4)

        for t in range(12):
            score = test_score[t]
            print(f"T{t:02d} {score:.4f}|", end="")

        print(f"score_mean: {score_mean}", end="")
        print()

    return test_score

def main(args):

    torch.cuda.set_device(args['gpu'])
    set_random_seed(args['random_seed'])
    
    dataset, train_set, val_set, test_set = load_dataset_for_classification(args)  # 6264, 783, 784

    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
   
    if args['pre_trained']:
        args['num_epochs'] = 0
        model = load_pretrained(args['exp'])
    else:
        args['n_tasks'] = dataset.n_tasks 
        if args['method'] == 'twp':
            model = load_mymodel(args)
            print(model)
        else:
            model = load_model(args)
            for name, parameters in model.named_parameters():
                print(name, ':', parameters.size())
        
        method = args['method']
        life_model = importlib.import_module(f'LifeModel.{method}_model')
        life_model_ins = life_model.NET(model, args)
        data_loader = DataLoader(train_set, batch_size=len(train_set),
            collate_fn=collate_molgraphs, shuffle=True)
        life_model_ins.data_loader = data_loader

        loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights.cuda(),
                                           reduction='none')


    model.cuda()
    score_mean = []
    score_matrix = np.zeros([args['n_tasks'], args['n_tasks']])

    prev_model = None
    for task_i in range(12):
        print('\n********'+ str(task_i))
        stopper = EarlyStopping(patience=args['patience'])
        for epoch in range(args['num_epochs']):
            # Train
            if args['method'] == 'lwf':
                life_model_ins.observe(train_loader, loss_criterion, task_i, args, prev_model)
            else:
                life_model_ins.observe(train_loader, loss_criterion, task_i, args)

            # Validation and early stop
            val_score = run_an_eval_epoch(args, model, val_loader, task_i)
            early_stop = stopper.step(val_score, model)
            
            if early_stop:
                print(epoch)
                break

        if not args['pre_trained']:
            stopper.load_checkpoint(model)

        score_matrix[task_i] = run_eval_epoch(args, model, test_loader)
        prev_model = copy.deepcopy(life_model_ins).cuda()

    print('AP: ', round(np.mean(score_matrix[-1,:]),4))
    backward = []
    for t in range(args['n_tasks']-1):
        b = score_matrix[args['n_tasks']-1][t]-score_matrix[t][t]
        backward.append(round(b, 4))
    mean_backward = round(np.mean(backward),4)        
    print('AF: ', mean_backward)
    

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='MoleculeNet')
    parser.add_argument('-m', '--model', type=str, choices=['GCN', 'GAT', 'Weave'],
                        help='Model to use')
    parser.add_argument('-e', '--method', type=str, choices=['finetune', 'lwf', 'gem', 'ewc', 'mas', 'twp', 'jointtrain'], 
                        default='single', help='Method to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['Tox21'], default='Tox21',
                        help='Dataset to use')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")

    # ewc/mas/gem
    parser.add_argument('-me', '--memory-strength', type=float, default=10000,
                        help="memory strength, 10000 for ewc/mas, 0.5 for gem")
    # gem
    parser.add_argument('-n', '--n-memories', type=int, default=100,
                        help="number of memories")

    # parameters for our method (twp)
    parser.add_argument('-l', '--lambda_l', type=float, default=10000, 
                        help=" ")    
    parser.add_argument('-t', '--lambda_t', type=float, default=100, 
                        help=" ")    
    parser.add_argument('-b', '--beta', type=float, default=0.1, 
                        help=" ")

    parser.add_argument('-s', '--random_seed', type=int, default=0,
                        help="seed for exp")
    
    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)