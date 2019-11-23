import pandas as pd
from torch.utils.data import DataLoader
from data.dataset import MovieReviewSet
from utils.config import dataset_config_from_file

config = dataset_config_from_file("conf/dataset.json")


def split(tsv, limit=None):
    raw_df = pd.read_csv(tsv, sep='\t')
    if limit:
        raw_df = raw_df.head(limit)
    count = raw_df.shape[0]
    raw_df = raw_df.sample(frac=1).reset_index()
    n_train = int(count * 0.8)
    n_evaluate = int(count * 0.1)
    train_df = raw_df.iloc[:n_train].reset_index()
    evaluate_df = raw_df.iloc[n_train:n_train+n_evaluate].reset_index()
    test_df = raw_df.iloc[n_train+n_evaluate:].reset_index()
    return MovieReviewSet(train_df), MovieReviewSet(evaluate_df), MovieReviewSet(test_df)


train_dataset, evaluate_dataset, test_dataset = split(config['path'], limit=config['limit'])
batch_size = config['batch_size']
train_dataloader = DataLoader(train_dataset, batch_size=(batch_size['train'] or len(train_dataset)))
evaluate_dataloader = DataLoader(evaluate_dataset, batch_size=(batch_size['evaluate'] or len(evaluate_dataset)))
test_dataloader = DataLoader(test_dataset, batch_size=(batch_size['test'] or len(test_dataset)))
