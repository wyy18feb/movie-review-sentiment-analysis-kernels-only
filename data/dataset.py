import torch
from torch.utils.data import Dataset
from models.tokenizer import tokenizer


class Transform(object):
    max_seq_length = 64

    def __call__(self, sample):
        x, y = sample
        x = tokenizer.encode(x, add_special_tokens=True)
        if len(x) > self.max_seq_length:
            x = x[:self.max_seq_length]
        x += [0] * (self.max_seq_length-len(x))
        assert len(x) == self.max_seq_length
        input_tensor = torch.tensor(x)
        output_tensor = torch.tensor(y)
        sample = input_tensor, output_tensor
        return sample


class MovieReviewSet(Dataset):
    def __init__(self, df, transform=Transform()):
        self.df = df
        self.input = df['Phrase']
        self.output = df['Sentiment']
        self.len = df.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.input[index], self.output[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len
