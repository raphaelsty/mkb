import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_json

__all__ = ['Wn18rr']


class Wn18rr(Fetch):
    """Wn18rr dataset.

    Wn18rr aim to iterate over the associated dataset. It provide positive samples, corresponding
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

        >>> from kdmkr import datasets

        >>> wn18rr = datasets.Wn18rr(batch_size=1, shuffle=True, seed=42)

        >>> wn18rr
        Wn18rr dataset
            Batch size          1
            Entities            40923
            Relations           11
            Shuffle             True
            Train triples       86834
            Validation triples  3033
            Test triples        3134


        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(wn18rr)
        ...     print(positive_sample, weight, mode)
        tensor([[2699,    4, 2010]]) tensor([0.1622]) tail-batch
        tensor([[ 9667,     5, 15434]]) tensor([0.1302]) head-batch
        tensor([[ 9023,     0, 25815]]) tensor([0.2357]) tail-batch

    References:
        1. [Dettmers, Tim, et al. "Convolutional 2d knowledge graph embeddings." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.](https://arxiv.org/pdf/1707.01476.pdf)

    """

    def __init__(self, batch_size, shuffle=False, num_workers=1, seed=None):

        self.filename = 'wn18rr'

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f'{path}/train.csv'),
            valid=read_csv(file_path=f'{path}/valid.csv'),
            test=read_csv(file_path=f'{path}/test.csv'),
            entities=read_json(f'{path}/entities.json'),
            relations=read_json(f'{path}/relations.json'),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed
        )
