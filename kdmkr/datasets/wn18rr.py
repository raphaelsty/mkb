import json
import os

from ..stream import fetch_dataset
from ..utils import read_csv

__all__ = ['WN18RR']


class WN18RR(fetch_dataset.FetchDataset):
    """Iter over WN18RR

    Example:

        :
            >>> from kdmkr import datasets
            >>> import torch

            >>> wn18rr = datasets.WN18RR(batch_size=1, negative_sample_size=1, shuffle=True,
            ... seed=42)

            >>> _ = torch.manual_seed(42)

            >>> for _ in range(3):
            ...     positive_sample, negative_sample, weight, mode = next(wn18rr)
            ...     print(positive_sample, negative_sample, weight, mode)
            tensor([[2699,    4, 2010]]) tensor([[15795]]) tensor([0.1622]) tail-batch
            tensor([[ 9667,     5, 15434]]) tensor([[15795]]) tensor([0.1302]) head-batch
            tensor([[ 9023,     0, 25815]]) tensor([[38158]]) tensor([0.2357]) tail-batch

    """
    def __init__(self, batch_size, negative_sample_size=1024, shuffle=False, num_workers=1,
        seed=None):

        self.folder    = 'wn18rr'
        self.directory = f'{os.path.dirname(os.path.realpath(__file__))}/{self.folder}'

        self.train_file_path = f'{self.directory}/train.csv'
        self.valid_file_path = f'{self.directory}/valid.csv'
        self.test_file_path  = f'{self.directory}/test.csv'

        self.entities_file_path  = f'{self.directory}/entities.json'
        self.relations_file_path = f'{self.directory}/relations.json'

        super().__init__(train=read_csv(file_path=self.train_file_path),
            valid=read_csv(file_path=self.valid_file_path), test=read_csv(file_path=self.test_file_path),
            batch_size=batch_size, negative_sample_size=negative_sample_size, shuffle=shuffle,
            num_workers=num_workers, seed=seed,
            entities=json.loads(open(self.entities_file_path).read()),
            relations = json.loads(open(self.relations_file_path).read()),
        )
