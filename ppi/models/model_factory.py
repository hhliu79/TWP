from .gat import GAT
import torch.nn.functional as F


def get_model(args, dataset, model_name="GAT"):
    
    train_g, train_dataloader, valid_dataloader, test_dataloader, n_classes, num_feats, num_train, num_valid, num_test = dataset  

    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(train_g,
                args.num_layers,
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