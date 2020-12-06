import torch
import torch.nn as nn
import dgl.function as fn
#from dgl.nn.pytorch import edge_softmax, GATConv
from .gatconv import GATConv
from .layers import PairNorm
import dgl
import tqdm

class GAT(nn.Module):
    def __init__(self,
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
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(
            (in_dim, in_dim), num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None))
        self.norm_layers.append(PairNorm())
        self.n_classes = num_classes
        self.n_hidden = num_hidden
        self.heads = heads
        self.n_known = 5
        
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            self.norm_layers.append(PairNorm())
        
        self.gat_layers.append(GATConv(
            (num_hidden * heads[-2],num_hidden * heads[-2]), num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
            

    def forward(self, blocks, inputs):
        h = inputs # torch.Size([106373, 602])
        '''
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.activation(h)
        logits = self.gat_layers[-1](g, h).mean(1)
        
        '''    
        elist = []
        for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):

            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            
            h_dst = h[:block.number_of_dst_nodes()]  #torch.Size([9687, 602])
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h, e = layer(block, (h,h_dst))
            elist.append(e)
            
            if l != len(self.gat_layers) - 1:
                h = h.flatten(1)
                h = self.activation(h)
               # h = self.dropout(h)
            else:
                h = h.mean(1)
        return h, elist
    
    
    def inference(self, g, x, batch_size):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = torch.arange(g.number_of_nodes())
        for l, layer in enumerate(self.gat_layers):
            y = torch.zeros(g.number_of_nodes(), (self.heads[l] * self.n_hidden) if l != len(self.gat_layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].cuda()
                h_dst = h[:block.number_of_dst_nodes()]
                h, _ = layer(block, (h, h_dst))
                if l != len(self.gat_layers) - 1:
                    h = h.flatten(1)
                    h = self.activation(h)
                else:
                    h = h.mean(1)

                y[start:end] = h.cpu()

            x = y
        return y
