import csv
import json
import os
import zipfile

from ..stream import fetch_dataset


__all__ = ['WN18RR']


class WN18RR(fetch_dataset.FetchDataset):
    """Iter over WN18RR

    Example:

        :
            >>> from kdmkr import datasets

            >>> wn18rr = datasets.WN18RR(batch_size=1, negative_sample_size=1, seed=42)

            >>> for _ in range(3):
            ...     positive_sample, negative_sample, weight, mode = next(wn18rr)
            ...     print(positive_sample, negative_sample, weight, mode)
            tensor([[    0,     0, 10211]]) tensor([[15795]]) tensor([0.3536]) tail-batch
            tensor([[    0,     0, 10211]]) tensor([[15795]]) tensor([0.3536]) head-batch
            tensor([[   1,    1, 8949]]) tensor([[38158]]) tensor([0.3162]) tail-batch

    """
    def __init__(self, batch_size, negative_sample_size=1024, shuffle=False, num_workers=1,
        seed=None):

        self.directory = os.path.dirname(os.path.realpath(__file__))

        super().__init__(train=self.read_csv(file='train.csv'),
            valid=self.read_csv(file='valid.csv'), test=self.read_csv(file='test.csv'),
            batch_size=batch_size, negative_sample_size=negative_sample_size, shuffle=shuffle,
            num_workers=num_workers, seed=seed,
            entities=json.loads(open(f'{self.directory}/wn18rr/entities.json').read()),
            relations = json.loads(open(f'{self.directory}/wn18rr/relations.json').read()),
        )

    def read_csv(self, file):
        with open(f'{self.directory}/wn18rr/{file}', 'r') as csv_file:
            return [(int(head), int(relation), int(tail))
                for head, relation, tail in csv.reader(csv_file)]
