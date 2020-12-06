import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog
from .gem_utils import store_grad, overwrite_grad, project2cone2

class NET(nn.Module):
    """
    wrap the base_model to be a lifelong model
    """    
    def __init__(self,
                model,
                task_manager,
                args):
        super(NET, self).__init__()        
        self.net = model
        self.task_manager = task_manager

        self.ce = nn.CrossEntropyLoss()
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()
        self.opt = optim.Adam(self.net.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        
        self.margin = args.memory_strength
        self.n_memories = args.n_memories

        # allocate episodic memory
        # for semi-supervised data, it will store the training mask for every old tasks
        self.memory_data = {}
        
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.task_manager.n_tasks).cuda()
        
        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.n_batch = 0
        self.old_epoch = -1
    
    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, g, features, labels, t, train_mask, epoch):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            self.memory_data[t] = []
        
        if epoch != self.old_epoch:
            self.n_batch = 0
            self.old_epoch = epoch

        self.net.g = g
        for layer in self.net.gat_layers:
            layer.g = g
            
        # Update ring buffer storing examples from current task
        if epoch == 0:
            tmask = torch.zeros_like(train_mask)
            n = 0
            for i in range(train_mask.shape[0]):
                if train_mask[i]:
                    tmask[i] = train_mask[i]
                    n = n + 1
                if n == self.n_memories:
                    break
            self.memory_data[t].append(tmask)
        
        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            self.net.zero_grad()
            # fwd/bwd on the examples in the memory
            
            offset1, offset2 = self.task_manager.get_label_offset(old_task_i)
            output, _ = self.net(features.float())
            try:
                old_task_loss = self.loss_fcn(
                                output[self.memory_data[old_task_i][self.n_batch], offset1: offset2],
                                labels[self.memory_data[old_task_i][self.n_batch], offset1: offset2].float())
                old_task_loss.backward()
                store_grad(self.net.parameters, self.grads, self.grad_dims,
                            old_task_i)
            except:
                1

        # now compute the grad on the current minibatch
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output, _  = self.net(features.float())
        loss = self.loss_fcn(output[train_mask, offset1: offset2], labels[train_mask, offset1: offset2].float())
        loss.backward()
        self.n_batch += 1

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.net.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.net.parameters, self.grads[:, t],
                               self.grad_dims)
        
        self.opt.step()
        return loss