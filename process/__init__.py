import torch


def accuracy(yhat, y):
    num_correct = torch.eq(yhat, y).sum().float()
    return num_correct / torch.numel(y), num_correct, torch.numel(y)


from process._train import train
from process._evaluate import evaluate
