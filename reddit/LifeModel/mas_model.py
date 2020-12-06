import torch
import dgl


class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()
        self.reg = args.memory_strength

        self.task_manager = task_manager

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = 0
        self.optpar = []
        self.fisher = []
        self.mem_mask = None

        self.n_memories = args.n_memories

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, blocks, features, labels, t):
        self.net.train()
        
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        labels_of_current_task = self.task_manager.labels_of_tasks[t]        
        train_mask = torch.BoolTensor( [l in labels_of_current_task for l in labels.detach().cpu()] )        
        output, _ = self.net(blocks, features)        
        loss = self.ce(output[train_mask, offset1: offset2], labels[train_mask] - offset1)
        
        if t > 0:
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()
                
        loss.backward()
        self.opt.step()
        
        return loss
    
    
    def compute_gradient(self, blocks, features, labels, t):
        self.net.zero_grad()
        
        self.optpar = []
        self.fisher = []

        offset1, offset2 = self.task_manager.get_label_offset(t)
        labels_of_current_task = self.task_manager.labels_of_tasks[t]        
        train_mask = torch.BoolTensor( [l in labels_of_current_task for l in labels.detach().cpu()] )        
        output, _ = self.net(blocks, features)
        output = output[train_mask, offset1: offset2]
    
        output.pow_(2)
        loss = output.mean()
        self.net.zero_grad()
        loss.backward()
            
        for p in self.net.parameters():
            pd = p.data.clone()
            pg = p.grad.data.clone().pow(2)
            self.fisher.append(pg)
            self.optpar.append(pd)
            
        self.current_task = t + 1
    