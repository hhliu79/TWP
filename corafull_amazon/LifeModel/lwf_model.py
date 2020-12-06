import torch
import copy
import os
from torch.autograd import Variable
from models.utils import EarlyStopping, evaluate, accuracy
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn as nn


def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True).cuda()


def kaiming_normal_init(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, torch.nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
        
class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager
        self.heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        self.args = args
        self.activation = F.elu

        # setup network
        self.net = model
        self.net.apply(kaiming_normal_init)                
        self.feature_extractor = self.net.gat_layers[0]
        self.gat = self.net.gat_layers[-1]
        
        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()

        self.current_task = 0
        self.n_classes = 5
        self.n_known = 0
        self.prev_model = None
 
    def forward(self, features):
        
        h = features
        g = self.task_manager.g
        h = self.feature_extractor(g, h)[0].flatten(1)
        h = self.activation(h) 
        h = self.gat(g, h)[0].mean(1) 
        return h      
                
    def observe(self, features, labels, t, train_mask, prev_model):
        self.net.train()   
        # if new task
        if t != self.current_task:            
            self.current_task = t        
            
        self.net.zero_grad()
        self.cuda()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        logits = self.net(features)  
        if isinstance(logits,tuple):
            logits = logits[0]
        logits = logits[train_mask]

        logits_cls = logits[:, offset1:offset2]
        labels = labels[train_mask] - offset1  
        loss = self.ce(logits_cls, labels)
        
        if t > 0:
            target = prev_model.forward(features)[train_mask]
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt)  
                logits_dist = logits[:,o1:o2]
                dist_target = target[:,o1:o2]
                dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
                loss = loss + dist_loss
        
        loss.backward()
        self.opt.step()
    