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
        self.observed_masks = []

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, features, labels, t, train_mask):
        self.net.train()
        
        if t != self.current_task:
            self.observed_masks.append(train_mask)
            self.current_task = t

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output, _ = self.net(features)
        loss = self.ce((output[train_mask, offset1: offset2]),
                            labels[train_mask] - offset1)
        
        for old_t, mask in enumerate(self.observed_masks[:-1]):
            offset1, offset2 = self.task_manager.get_label_offset(old_t)
            output, _ = self.net(features)
            loss_aux= self.ce((output[mask, offset1: offset2]),
                            labels[mask] - offset1)
            loss = loss + loss_aux

        loss.backward()
        self.opt.step()
