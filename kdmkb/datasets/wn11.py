import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_json


__all__ = ['Wn11']


class Wn11(Fetch):
    """Wn11 dataset.

    Wn11 aim to iterate over the associated dataset. It provide positive samples, corresponding
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

        >>> dataset = datasets.Wn11(batch_size=1, shuffle=True, seed=42)

        >>> dataset
        Wn11 dataset
            Batch size         1
            Entities           38551
            Relations          11
            Shuffle            True
            Train triples      112581
            Validation triples 2609
            Test triples       10544


        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(dataset)
        ...     print(positive_sample, weight, mode)
        tensor([[29798,     2, 16107]]) tensor([0.3162]) tail-batch
        tensor([[36732,     2,   563]]) tensor([0.3333]) head-batch
        tensor([[ 4115,     0, 20415]]) tensor([0.3015]) tail-batch


    References:
        1. [An Open-source Framework for Knowledge Embedding implemented with PyTorch.](https://github.com/thunlp/OpenKE)

    """

    def __init__(self, batch_size, shuffle=False, num_workers=1, seed=None):

        self.filename = 'wn11'

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f'{path}/train.csv'),
            valid=read_csv(file_path=f'{path}/valid.csv'),
            test=read_csv(file_path=f'{path}/test.csv'),
            entities=read_json(f'{path}/entities.json'),
            relations=read_json(f'{path}/relations.json'),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed
        )
