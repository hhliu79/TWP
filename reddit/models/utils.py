import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn import metrics

def calc_f1(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return metrics.f1_score(y_true, y_pred, average="micro")

def evaluate(model, g, inputs, labels, mask, batch_size, label_offset1, label_offset2):
    with torch.no_grad():
        model.eval()
        pred = model.inference(g, inputs, batch_size)[:, label_offset1 : label_offset2]
        pred = pred[mask]
        labels = labels[mask] - label_offset1
        score = calc_f1(labels.data.cpu().numpy(), pred)
        return score