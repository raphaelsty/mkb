import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_json


__all__ = ['Fb15k']


class Fb15k(Fetch):
    """Fb15k dataset.

    Fb15k aim to iterate over the associated dataset. It provide positive samples, corresponding
    weights and the mode (head batch / tail batch)

    Parameters:
        batch_size (int): Size of the batch.
        shuffle (bool): Whether to shuffle the dataset or not.
        num_workers (int): Number of workers dedicated to iterate on the dataset.
        seed (int): Random state.

    Attributes:
        train (list): Training set.
        valid (list): Validation set.
        test (list): Testing set.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.

    Example:

        >>> from kdmkb import datasets

        >>> dataset = datasets.Fb15k(batch_size=1, shuffle=True, seed=42)

        >>> dataset
        Fb15k dataset
            Batch size         1
            Entities           14951
            Relations          1345
            Shuffle            True
            Train triples      483142
            Validation triples 50000
            Test triples       59071

        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(dataset)
        ...     print(positive_sample, weight, mode)
        tensor([[1548,  247, 7753]]) tensor([0.1043]) tail-batch
        tensor([[4625,   53, 6478]]) tensor([0.2425]) head-batch
        tensor([[135,  52, 170]]) tensor([0.0902]) tail-batch


    References:
        1. [An Open-source Framework for Knowledge Embedding implemented with PyTorch.](https://github.com/thunlp/OpenKE)

    """

    def __init__(self, batch_size, shuffle=False, num_workers=1, seed=None):

        self.filename = 'fb15k'

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f'{path}/train.csv'),
            valid=read_csv(file_path=f'{path}/valid.csv'),
            test=read_csv(file_path=f'{path}/test.csv'),
            entities=read_json(f'{path}/entities.json'),
            relations=read_json(f'{path}/relations.json'),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed
        )
