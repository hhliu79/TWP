import torch
import copy
import os
from torch.autograd import Variable
import torch.nn.functional as F

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
        #self.feature_extractor = torch.nn.Sequential(*list(self.net.gat_layers[:-1]))
        self.feature_extractor = self.net.gat_layers[0]
        
        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()
        # setup memories
        self.current_task = 0
        self.n_classes = 5
        self.n_known = 0
 
        self.new_classes = task_manager.n_labels_per_task
        self.num_layers = args.num_layers
    
    def forward(self, blocks, features):
        
        h = features
        for l, (layer, block) in enumerate(zip(self.net.gat_layers, blocks)):
            
            h_dst = h[:block.number_of_dst_nodes()]  #
            if l != self.num_layers:
                h, _ = self.feature_extractor(block, (h,h_dst))
                h = h.flatten(1)
                h = self.activation(h)
            else:
                h, _ = self.gat(block, (h,h_dst))
                h = h.mean(1)
        return h
    
    
    def observe(self, blocks, features, labels, t, prev_model):
        
        if t != self.current_task:      
            self.current_task = t      
                        
        self.net.zero_grad()        
        offset1, offset2 = self.task_manager.get_label_offset(t)
        labels_of_current_task = self.task_manager.labels_of_tasks[t]        
        train_mask = torch.BoolTensor( [l in labels_of_current_task for l in labels.detach().cpu()] )
        
        output, _ = self.net(blocks, features)
        output = output[train_mask]  
        
        logits_cls = output[:, offset1:offset2]   
        labels = labels[train_mask] - offset1  
    
        loss = self.ce(logits_cls, labels)
        
        if t > 0:
            target = prev_model.forward(blocks, features)
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt)
                logits_dist = output[:,o1:o2]
                dist_target = target[train_mask, o1:o2]  
                dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 4)
                loss = loss + dist_loss 
        
        loss.backward()
        self.opt.step()
        
        return loss
        
        