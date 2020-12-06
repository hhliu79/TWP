import numpy as np
import torch
from sklearn.metrics import f1_score

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, mask, label_offset1, label_offset2):
    model.eval()
    with torch.no_grad():
        output, _ = model(features)
        logits = output[:, label_offset1 : label_offset2]
        logits = logits[mask]
        labels = labels[mask] - label_offset1
        return accuracy(logits, labels)
