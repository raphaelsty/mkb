import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_json


__all__ = ['CountriesS2']


class CountriesS2(Fetch):
    """CountriesS2 dataset.

    countriesS2 aim to iterate over the associated dataset. It provide positive samples, corresponding
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

        >>> countries = datasets.CountriesS2(batch_size=1, shuffle=True, seed=42)

        >>> countries
        CountriesS2 dataset
            Batch size         1
            Entities           271
            Relations          2
            Shuffle            True
            Train triples      1063
            Validation triples 24
            Test triples       24

        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(countries)
        ...     print(positive_sample, weight, mode)
        tensor([[229,   0, 269]]) tensor([0.1601]) tail-batch
        tensor([[216,   1, 246]]) tensor([0.2041]) head-batch
        tensor([[47,  0, 72]]) tensor([0.2294]) tail-batch


    References:
        1. [Bouchard, Guillaume, Sameer Singh, and Theo Trouillon. "On approximate reasoning capabilities of low-rank vector spaces." 2015 AAAI Spring Symposium Series. 2015.](https://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10257/10026)
        2. [Datasets for Knowledge Graph Completion with Textual Information about Entities](https://github.com/villmow/datasets_knowledge_embedding)

    """

    def __init__(self, batch_size, shuffle=False, num_workers=1, seed=None):

        self.filename = 'countries_s2'

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f'{path}/train.csv'),
            valid=read_csv(file_path=f'{path}/valid.csv'),
            test=read_csv(file_path=f'{path}/test.csv'),
            entities=read_json(f'{path}/entities.json'),
            relations=read_json(f'{path}/relations.json'),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed
        )
