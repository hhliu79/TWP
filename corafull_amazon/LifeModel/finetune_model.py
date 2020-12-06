import torch


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
        self.current_task = 0

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, features, labels, t, train_mask):
        self.net.train()
        
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
    
        output = self.net(features)
        if isinstance(output,tuple):
            output = output[0]
        loss = self.ce((output[train_mask, offset1: offset2]),
                            labels[train_mask] - offset1)
        
        loss.backward()
        self.opt.step()

