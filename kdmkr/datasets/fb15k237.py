import csv
import json
import os
import zipfile

from ..stream import fetch_dataset


__all__ = ['FB15K237']


class FB15K237(fetch_dataset.FetchDataset):
    """Iter over FB15K237

    Example:

        :
            >>> from kdmkr import datasets
            >>> import torch

            >>> fb15k237 = datasets.FB15K237(batch_size=1, negative_sample_size=1, shuffle=True,
            ... seed=42)

            >>> _ = torch.manual_seed(42)

            >>> for _ in range(3):
            ...     positive_sample, negative_sample, weight, mode = next(fb15k237)
            ...     print(positive_sample, negative_sample, weight, mode)
            tensor([[11775,    30, 11576]]) tensor([[7270]]) tensor([0.2887]) tail-batch
            tensor([[ 203,   21, 5251]]) tensor([[7270]]) tensor([0.3333]) head-batch
            tensor([[4994,   31, 8256]]) tensor([[13418]]) tensor([0.0343]) tail-batch

    """
    def __init__(self, batch_size, negative_sample_size=1024, shuffle=False, num_workers=1,
        seed=None):

        self.directory = os.path.dirname(os.path.realpath(__file__))

        super().__init__(train=self.read_csv(file='train.csv'),
            valid=self.read_csv(file='valid.csv'), test=self.read_csv(file='test.csv'),
            batch_size=batch_size, negative_sample_size=negative_sample_size, shuffle=shuffle,
            num_workers=num_workers, seed=seed,
            entities=json.loads(open(f'{self.directory}/fb15k237/entities.json').read()),
            relations = json.loads(open(f'{self.directory}/fb15k237/relations.json').read()),
        )

    def read_csv(self, file):
        with open(f'{self.directory}/fb15k237/{file}', 'r') as csv_file:
            return [(int(head), int(relation), int(tail))
                for head, relation, tail in csv.reader(csv_file)]
