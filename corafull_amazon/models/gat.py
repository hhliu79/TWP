import torch
import torch.nn as nn
import dgl.function as fn
from .gatconv import GATConv
from .layers import PairNorm


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None))
        # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[0]))
        self.norm_layers.append(PairNorm())
        
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[l]))
            self.norm_layers.append(PairNorm())
        # output projection

        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        e_list = []
        
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](self.g, h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
            
        # output projection
        logits, e = self.gat_layers[-1](self.g, h)
        logits = logits.mean(1)
        e_list = e_list + e
        
        return logits, e_list