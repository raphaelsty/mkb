import csv
import json
import os
import zipfile

from ..stream import fetch_dataset


__all__ = ['KDMKRWN18RR']


class KDMKRWN18RR(fetch_dataset.FetchDataset):
    """Iter over KDMKRWN18RR

    Example:

        :
            >>> from kdmkr import datasets
            >>> import torch

            >>> kdmkr_wn18rr = datasets.KDMKRWN18RR(batch_size=1, negative_sample_size=1, shuffle=True,
            ... seed=42)

            >>> _ = torch.manual_seed(42)

            >>> for _ in range(3):
            ...     positive_sample, negative_sample, weight, mode = next(kdmkr_wn18rr)
            ...     print(positive_sample, negative_sample, weight, mode)
            tensor([[2699,    4, 2010]]) tensor([[15795]]) tensor([0.1622]) tail-batch
            tensor([[ 9667,     5, 15434]]) tensor([[15795]]) tensor([0.1302]) head-batch
            tensor([[ 9023,     0, 25815]]) tensor([[38158]]) tensor([0.2357]) tail-batch

    """
    def __init__(self, batch_size, negative_sample_size=1024, shuffle=False, num_workers=1,
        seed=None):

        self.directory = os.path.dirname(os.path.realpath(__file__))

        super().__init__(train=self.read_csv(file='train.csv'),
            valid=self.read_csv(file='valid.csv'), test=self.read_csv(file='test.csv'),
            batch_size=batch_size, negative_sample_size=negative_sample_size, shuffle=shuffle,
            num_workers=num_workers, seed=seed,
            entities=json.loads(open(f'{self.directory}/kdmkr_wn18rr/entities.json').read()),
            relations = json.loads(open(f'{self.directory}/kdmkr_wn18rr/relations.json').read()),
        )

        # We just add relation to distill knwoledge
        self.n_relation = 11

    def read_csv(self, file):
        with open(f'{self.directory}/kdmkr_wn18rr/{file}', 'r') as csv_file:
            return [(int(head), int(relation), int(tail))
                for head, relation, tail in csv.reader(csv_file)]
