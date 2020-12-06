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

        self.current_task = 0
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}
        self.data_loader = None

        self.lambda_l = args['lambda_l']
        self.lambda_t = args['lambda_t']
        self.beta = args['beta']


    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, data_loader, loss_criterion, task_i, args):

        self.net.train()

        if task_i != self.current_task:
            self.optimizer.zero_grad()

            self.fisher_loss[self.current_task] = [] 
            self.fisher_att[self.current_task] = []  
            self.optpar[self.current_task] = []

            for batch_id, batch_data in enumerate(self.data_loader):
                smiles, bg, labels, masks = batch_data
                labels, masks = labels.cuda(), masks.cuda()
                logits, elist = predict(args, self.net, bg)

                loss = loss_criterion(logits, labels) * (masks != 0).float()
                loss = loss[:,self.current_task].mean()
                loss.backward(retain_graph=True) 

                for p in self.net.parameters():
                    pd = p.data.clone()
                    try:
                        pg = p.grad.data.clone().pow(2)
                        self.fisher_loss[self.current_task].append(pg)
                        self.optpar[self.current_task].append(pd)
                    except:
                        1

                if isinstance(elist,list):
                    eloss = torch.norm(elist[0])
                else:
                    eloss = elist
                eloss.backward()
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2) 
                        self.fisher_att[self.current_task].append(pg)   
                    except:
                        1
                self.current_task = task_i

        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.cuda(), masks.cuda()
            logits, elist = predict(args, self.net, bg)

            # Mask non-existing labels
            loss = loss_criterion(logits, labels) * (masks != 0).float()
            loss = loss[:,task_i].mean()

            loss.backward(retain_graph=True)
            grad_norm = 0
            for p in self.net.parameters():
                try:
                    pg = p.grad.data.clone()
                    grad_norm += torch.norm(pg, p=1)  
                except:
                    1 

            for tt in range(task_i):
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)  # error
                        l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                        l = l * (p - self.optpar[tt][i]).pow(2)
                        loss += l.sum()    
                        i += 1
                    except:
                        1

            loss = loss + self.beta * grad_norm 
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)
        
        train_score = np.mean(train_meter.compute_metric(args['metric_name']))