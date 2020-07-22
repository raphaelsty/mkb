import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_json


__all__ = ['Fb15k237']


class Fb15k237(Fetch):
    """Fb15k237 dataset.

    Fb15k237 aim to iterate over the associated dataset. It provide positive samples, corresponding
    weights and the mode (head batch / tail batch)

    Parameters:
        batch_size (int): Number of sample to iter on.
        shuffle (bool): Wether to shuffle the dataset or not.
        num_workers (int): Number of workers dedicated to iterate on the dataset.
        seed (int): Random seed.

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

        >>> fb15k237 = datasets.Fb15k237(batch_size=1, shuffle=True, seed=42)

        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(fb15k237)
        ...     print(positive_sample, weight, mode)
        tensor([[5222,   24, 1165]]) tensor([0.2887]) tail-batch
        tensor([[8615,   12, 2350]]) tensor([0.3333]) head-batch
        tensor([[  16,   15, 4726]]) tensor([0.0343]) tail-batch

    References:
        1. [Toutanova, Kristina, et al. "Representing text for joint embedding of text and knowledge bases." Proceedings of the 2015 conference on empirical methods in natural language processing. 2015.](https://www.aclweb.org/anthology/D15-1174.pdf)

    """

    def __init__(self, batch_size, shuffle=False, num_workers=1, seed=None):

        self.filename = 'fb15k237'

        self.directory = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f'{self.directory}/train.csv'),
            valid=read_csv(file_path=f'{self.directory}/valid.csv'),
            test=read_csv(file_path=f'{self.directory}/test.csv'),
            entities=read_json(f'{self.directory}/entities.json'),
            relations=read_json(f'{self.directory}/relations.json'),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed
        )
