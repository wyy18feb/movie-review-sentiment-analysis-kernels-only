import argparse
import torch
from data import train_dataloader
from process import train, evaluate
from models import model
from utils.log import logger


def main(args):
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.mode
    if mode == 'train':
        epochs = args.epochs
        evaluate_steps = args.evaluate_steps
        save_steps = args.save_steps
        save_dir = args.save_dir
        train(model, train_dataloader, epochs, evaluate_steps, save_steps, save_dir)
    elif mode == 'evaluate':
        model_path = args.model_path
        if model_path is None:
            logger.error('model_path is not set!')
            return
        model.load_state_dict(torch.load(model_path))
        model.eval()
        loss, acc = evaluate(model)
        logger.info("loss=%f, accuracy=%f", loss, acc)
    else:
        logger.error('wrong mode: %s', mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='train/evaluate')
    parser.add_argument('--epochs', type=int, default=3, help='epochs num when mode is set to "train"')
    parser.add_argument('--evaluate_steps', type=int, default=10, help='evaluate steps when mode is set to "train"')
    parser.add_argument('--save_steps', type=int, default=50, help='save steps when mode is set to "train"')
    parser.add_argument('--save_dir', type=str, default=".output", help='save dir when mode is set to "train"')
    parser.add_argument('--model_path', type=str, help='model path which is going to be evaluated')

    args = parser.parse_args()
    main(args)
