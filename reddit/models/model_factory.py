from .gat import GAT
import torch.nn.functional as F


def get_model(args, model_name="GAT"):
    
    #g, features, labels, train_mask, val_mask, test_mask = dataset

    #num_feats = features.shape[1]
    #n_classes = max(labels) + 1
    num_feats = 602
    n_classes = 41
    #n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual,
                args.method)
    return model