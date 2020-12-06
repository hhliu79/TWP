import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping, Meter
import numpy as np
from torch.autograd import Variable


def predict(args, model, bg):
    node_feats = bg.ndata[args['node_data_field']].cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata[args['edge_data_field']].cuda()
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)


def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True).cuda()

class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.optimizer = Adam(model.parameters(), lr=args['lr'])

    def forward(self, args, bg):
        logits = predict(args, self.net, bg)
        return logits

    def observe(self, data_loader, loss_criterion, task_i, args, prev_model):

        self.net.train()
        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # Mask non-existing labels
            loss = loss_criterion(logits, labels) * (masks != 0).float()
            loss = loss[:,task_i].mean()

            if task_i > 0:
                target = prev_model.forward(args, bg)
                for oldt in range(task_i):
                    logits_dist = torch.unsqueeze(logits[:,oldt], 0)
                    dist_target = torch.unsqueeze(target[:,oldt], 0)
                    dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
                    loss = loss + dist_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)
        
        train_score = np.mean(train_meter.compute_metric(args['metric_name']))
        