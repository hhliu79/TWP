import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping, Meter
import numpy as np
from torch.utils.data import DataLoader

def predict(args, model, bg):
    node_feats = bg.ndata.pop(args['node_data_field']).cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).cuda()
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)


class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.optimizer = Adam(model.parameters(), lr=args['lr'])

        # mas
        self.reg = args['memory_strength']
        self.current_task = 0
        self.optpar = []
        self.fisher = []
        self.data_loader = None

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, data_loader, loss_criterion, task_i, args):

        self.net.train()

        if task_i != self.current_task:
            self.optpar = []
            self.fisher = []
            self.optimizer.zero_grad()
            
            for batch_id, batch_data in enumerate(self.data_loader):
                smiles, bg, labels, masks = batch_data
                labels, masks = labels.cuda(), masks.cuda()
                output = predict(args, self.net, bg)[:,self.current_task]

                output.pow_(2)
                loss = output.mean()
                self.net.zero_grad()
                loss.backward()
    
                for p in self.net.parameters():
                    pd = p.data.clone()
                    try:
                        pg = p.grad.data.clone().pow(2)  # error
                        self.fisher.append(pg)
                        self.optpar.append(pd)
                    except:
                        1
                self.current_task = task_i


        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # Mask non-existing labels
            loss = loss_criterion(logits, labels) * (masks != 0).float()
            loss = loss[:,task_i].mean()

            if task_i > 0:
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2) 
                        l = self.reg * self.fisher[i]
                        l = l * (p - self.optpar[i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)
        
        train_score = np.mean(train_meter.compute_metric(args['metric_name']))