import torch
import copy


class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = -1
        self.observed_tasks = []

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, blocks, features, labels, t):
        self.net.train()
        
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t

        self.net.zero_grad()

        offset1, offset2 = self.task_manager.get_label_offset(t)
        labels_of_current_task = self.task_manager.labels_of_tasks[t]        
        train_mask = torch.BoolTensor( [l in labels_of_current_task for l in labels.detach().cpu()] )
        
        output, _  = self.net(blocks, features)
        loss = self.ce((output[train_mask, offset1: offset2]), labels[train_mask] - offset1)
        
        for old_task_i in self.observed_tasks[:-1]:
            o1, o2 = self.task_manager.get_label_offset(old_task_i)
            labels_of_task = self.task_manager.labels_of_tasks[old_task_i]     
            mask = torch.BoolTensor( [l in labels_of_task for l in labels.detach().cpu()] )
            
            output, _ = self.net(blocks,features)
            loss_aux= self.ce((output[mask, o1: o2]),
                            labels[mask] - o1)
            loss = loss + loss_aux

        loss.backward()
        self.opt.step()
        return loss
    