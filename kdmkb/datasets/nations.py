import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_json


__all__ = ['Nations']


class Nations(Fetch):
    """Nations dataset.

    Nations aim to iterate over the associated dataset. It provide positive samples, corresponding
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

        >>> nations = datasets.Nations(batch_size=1, shuffle=True, seed=42)

        >>> nations
        Nations dataset
            Batch size           1
            Entities            14
            Relations           56
            Shuffle           True
            Train triples     1619
            Validation triples 202
            Test triples       203


        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(nations)
        ...     print(positive_sample, weight, mode)
        tensor([[ 6,  5, 13]]) tensor([0.2085]) tail-batch
        tensor([[6, 0, 5]]) tensor([0.2182]) head-batch
        tensor([[10,  8,  7]]) tensor([0.2085]) tail-batch


    References:
        1. [Datasets for Knowledge Graph Completion with Textual Information about Entities](https://github.com/villmow/datasets_knowledge_embedding)

    """

    def __init__(self, batch_size, shuffle=False, num_workers=1, seed=None):

        self.filename = 'nations'

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f'{path}/train.csv'),
            valid=read_csv(file_path=f'{path}/valid.csv'),
            test=read_csv(file_path=f'{path}/test.csv'),
            entities=read_json(f'{path}/entities.json'),
            relations=read_json(f'{path}/relations.json'),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed
        )
