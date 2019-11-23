import torch
from torch import nn
from process import accuracy
from data import evaluate_dataloader
from utils.log import logger


def evaluate(model, dataloader=evaluate_dataloader):
    logger.debug('Evaluation started...')
    softmax = nn.Softmax(dim=1)
    model.eval()
    LOSS, CORRECT, TOTAL = 0, 0, 0
    for x, y in dataloader:
        loss, logits = model(x, labels=y)
        yhat = torch.argmax(softmax(logits), dim=1)
        acc, correct, total = accuracy(yhat, y)
        logger.debug("y   =%s", y)
        logger.debug("yhat=%s", yhat)
        logger.info("loss=%s, accuracy=%s", loss.item(), acc.item())
        LOSS += loss.item()
        CORRECT += correct
        TOTAL += total
    return LOSS, CORRECT/TOTAL
