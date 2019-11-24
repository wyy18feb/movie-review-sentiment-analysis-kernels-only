import os
import torch
from utils.log import logger
from models import optimizer
from process._evaluate import evaluate


def save(model, iteration, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = 'model_'+str(iteration)+'.pt'
    torch.save(model.state_dict(), os.path.join(save_dir, filename))
    logger.info('Saved %s', filename)


def train(model, dataloader, epochs, evaluate_steps, save_steps, save_dir):
    logger.debug('Prepared to train %d epochs in total', epochs)
    iterations = 0
    for epoch in range(epochs):
        logger.debug('Epoch %s started...', epoch+1)
        for x, y in dataloader:
            iterations += 1
            model.train()
            optimizer.zero_grad()
            loss, _ = model(x, labels=y)
            logger.debug('y=%s', y)
            loss.backward()
            optimizer.step()
            if evaluate_steps and iterations % evaluate_steps == 0:
                loss, acc = evaluate(model)
                logger.info('After %d iterations: loss=%f, accuracy=%f', iterations, loss, acc)
            if save_steps and iterations % save_steps == 0:
                save(model, iterations, save_dir)
    save(model, iterations, save_dir)
