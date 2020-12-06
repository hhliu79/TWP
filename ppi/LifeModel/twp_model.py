import torch
from torch.autograd import Variable
import numpy as np

class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager
        self.newg = task_manager.newg

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

        # setup memories
        self.current_task = 0
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}

        # hyper-parameters
        self.lambda_l = args.lambda_l
        self.lambda_t = args.lambda_t
        self.beta = args.beta

    def forward(self, features):
        output, elist = self.net(features)
        return output

    def observe(self, g, features, labels, t, train_mask):
        self.net.train()         

        self.net.g = g
        for layer in self.net.gat_layers:
            layer.g = g
            
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        
        output, elist = self.net(features.float())
        loss = self.loss_fcn((output[train_mask, offset1: offset2]), labels[train_mask, offset1:offset2].float())        
        loss.backward(retain_graph=True)
           
        grad_norm = 0
        for p in self.net.parameters():
            pg = p.grad.data.clone()               
            grad_norm += torch.norm(pg, p=1) 
        
        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()                     
        
        loss = loss + self.beta * grad_norm 
        loss.backward()
        self.opt.step()
        return loss
    
    
    def compute_gradient(self, g, features, labels, t, train_mask):
        
        self.net.zero_grad()        
        offset1, offset2 = self.task_manager.get_label_offset(t)
        
        self.net.g = g
        for layer in self.net.gat_layers:
            layer.g = g
            
        self.fisher_loss[t] = []
        self.optpar[t] = []
        self.fisher_att[t] = []
       
        output, elist = self.net(features.float())
        self.loss_fcn((output[train_mask, offset1: offset2]),
                      labels[train_mask, offset1: offset2].float()).backward(retain_graph=True)
                        
        for i, p in enumerate(self.net.parameters()):
            pd = p.data.clone()
            pg = p.grad.data.clone().pow(2)
            self.fisher_loss[t].append(pg)
            self.optpar[t].append(pd)

        eloss = torch.norm(elist[0]) 
        eloss.backward() 
            
        for i, p in enumerate(self.net.parameters()):
            pg = p.grad.data.clone().pow(2) 
            self.fisher_att[t].append(pg)
            
        self.current_task = t