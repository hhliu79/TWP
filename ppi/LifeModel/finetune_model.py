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
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

        # setup memories
        self.current_task = 0

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, g, features, labels, t, train_mask):
        self.net.train()
        
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        self.net.g = g
        for layer in self.net.gat_layers:
            layer.g = g
        output, _ = self.net(features.float())
        loss = self.loss_fcn(output[train_mask, offset1: offset2], labels[train_mask, offset1: offset2].float())
        
        loss.backward()
        self.opt.step()
        
        return loss
