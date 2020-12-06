import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax#, GATConv
from .gatconv import GATConv
from .layers import PairNorm
import dgl
import tqdm
import torch.nn.functional as F


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
                 residual,
                 method = None):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None))
        self.norm_layers.append(PairNorm())
        self.sigmoid = nn.Sigmoid()
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            self.norm_layers.append(PairNorm())
        
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))


    def forward(self, inputs):
        h = inputs 
        elist = []
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](self.g, h)
            h = h.flatten(1)
            h = self.activation(h)
            elist.append(e)
        
        logits, e = self.gat_layers[-1](self.g, h)
        logits = logits.mean(1)
        elist.append(e)
      
        return logits, elist