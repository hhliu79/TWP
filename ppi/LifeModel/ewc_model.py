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
        self.fisher = {}
        self.optpar = {}


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
        
        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()             
    
        loss.backward()
        self.opt.step()
        return loss
    
    
    def compute_gradient(self, g, features, labels, t, train_mask):
        
        self.net.zero_grad()

        offset1, offset2 = self.task_manager.get_label_offset(t)
        self.net.g = g
        for layer in self.net.gat_layers:
            layer.g = g
        self.fisher[t] = []
        self.optpar[t] = []        
        output, _ = self.net(features.float())

        print(output[train_mask, offset1: offset2].shape, labels[train_mask,].shape)
        self.loss_fcn((output[train_mask, offset1: offset2]),
                        labels[train_mask, offset1: offset2].float()).backward()                
        
        for p in self.net.parameters():
            pd = p.data.clone()
            pg = p.grad.data.clone().pow(2)
            self.fisher[t].append(pg)
            self.optpar[t].append(pd)
