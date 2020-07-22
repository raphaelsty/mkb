import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_json


__all__ = ['Fb15k237']


class Fb15k237(Fetch):
    """Iter over Fb15k237

    Example:

        :
            >>> from kdmkr import datasets
            >>> import torch

            >>> fb15k237 = datasets.Fb15k237(batch_size=1, shuffle=True, seed=42)

            >>> _ = torch.manual_seed(42)

            >>> for _ in range(3):
            ...     positive_sample, weight, mode = next(fb15k237)
            ...     print(positive_sample, weight, mode)
            tensor([[5222,   24, 1165]]) tensor([0.2887]) tail-batch
            tensor([[8615,   12, 2350]]) tensor([0.3333]) head-batch
            tensor([[  16,   15, 4726]]) tensor([0.0343]) tail-batch

    """
    def __init__(self, batch_size, shuffle=False, num_workers=1, seed=None):

        self.filename = 'fb15k237'

        self.directory = pathlib.Path(__file__).parent.joinpath(self.filename)

        self.train_file_path = f'{self.directory}/train.csv'
        self.valid_file_path = f'{self.directory}/valid.csv'
        self.test_file_path  = f'{self.directory}/test.csv'

        self.entities_file_path  = f'{self.directory}/entities.json'
        self.relations_file_path = f'{self.directory}/relations.json'

        super().__init__(train=read_csv(file_path=self.train_file_path),
            valid=read_csv(file_path=self.valid_file_path),
            test=read_csv(file_path=self.test_file_path),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed,
            entities=read_json(self.entities_file_path),
            relations=read_json(self.relations_file_path),
        )
