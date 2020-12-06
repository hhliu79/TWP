import torch


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
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

        # setup memories
        self.current_task = 0
        self.fisher = []
        self.optpar = []
        self.mem_mask = None

        self.n_memories = args.n_memories
        self.dataloader = None       

    def forward(self, features):
        output = self.net(features)
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
        
        if t > 0:
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()             
    
        loss.backward()
        self.opt.step()
        return loss
    
    
    def compute_gradient(self, g, features, labels, t, train_mask):
        
        self.net.zero_grad()
        self.optpar = []
        self.fisher = []
            
        offset1, offset2 = self.task_manager.get_label_offset(t)
        self.net.g = g
        for layer in self.net.gat_layers:
            layer.g = g       
        
        output, _ = self.net(features.float())
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