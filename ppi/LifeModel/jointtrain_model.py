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
        self.current_task = -1
        self.observed_masks = []

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
        output, _ = self.net(features.float())
        
        loss = self.loss_fcn((output[train_mask, offset1: offset2]),
                            labels[train_mask, offset1: offset2].float())

        for old_t in range(t):
            output, _ = self.net(features.float())

            offset1, offset2 = self.task_manager.get_label_offset(old_t)
            lbl = labels[:, offset1:offset2]
            lbl_sum = torch.sum(lbl, 1)                             
            mask = torch.BoolTensor( [l >= 0 for l in lbl_sum] )
            
            loss_aux= self.loss_fcn(output[mask, offset1: offset2], labels[mask, offset1: offset2].float())
            loss = loss + loss_aux

        loss.backward()
        self.opt.step()
        
        return loss
