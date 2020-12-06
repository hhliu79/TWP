import torch
import copy
import os
from torch.autograd import Variable
import torch.nn.functional as F
import random 
random.seed(1)

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
        self.gat = self.net.gat_layers[-1]
        self.feature_extractor = torch.nn.Sequential(*list(self.net.gat_layers[:-1]))
        
        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()
        # setup memories
        self.current_task = 0
        self.n_classes = 5
        self.n_known = 0
        self.prev_model = None
        self.new_classes = task_manager.n_labels_per_task
 
    
    def forward(self, g, features):
        
        h = features
        #g = self.task_manager.g
        for l in range(self.args.num_layers):
            h, e = self.feature_extractor[l](g, h)
            h = h.flatten(1)
            h = self.activation(h)      

        h, e = self.gat(g, h)
        h = h.mean(1)
        return h      

        
    def observe(self, g, features, labels, t, train_mask, prev_model):
        # if new task
        if t != self.current_task:            
            self.current_task = t        
            
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)  
        self.net.g = g
        for layer in self.net.gat_layers:
            layer.g = g
            
        output = self.net(features.float())  
        if isinstance(output,tuple):
            output = output[0]
            
        output = output[train_mask]
        logits_cls = output[:, offset1:offset2]
        labels = labels[train_mask, offset1: offset2]
        cls_loss = self.loss_fcn(logits_cls, labels.float())
        loss = cls_loss
        
        if t > 0:
            target = prev_model.forward(g, features.float())[train_mask]
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt)  
                logits_dist = output[:,o1:o2]
                dist_target = target[:,o1:o2]
                dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 4)
                loss = loss + dist_loss
        
        loss.backward()
        self.opt.step()
        
        return loss  