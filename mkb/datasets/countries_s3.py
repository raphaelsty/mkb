import os
import pathlib

from .dataset import Dataset

from ..utils import read_csv
from ..utils import read_json


__all__ = ['CountriesS3']


class CountriesS3(Dataset):
    """CountriesS3 dataset.

    countriesS3 aim to iterate over the associated dataset. It provide positive samples, corresponding
    weights and the mode (head batch / tail batch)

    Parameters:
        batch_size (int): Size of the batch.
        classification (bool): Must be set to True when using ConvE model to optimize BCELoss.
        shuffle (bool): Whether to shuffle the dataset or not.
        pre_compute (bool): Pre-compute parameters such as weights when using translationnal model
            (TransE, DistMult, RotatE, pRotatE, ComplEx). When using ConvE, pre-compute target
            matrix. When pre_compute is set to True, the model training is faster but it needs more
            memory.
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

        >>> from mkb import datasets

        >>> dataset = datasets.CountriesS3(batch_size=1, shuffle=True, seed=42)

        >>> dataset
        CountriesS3 dataset
            Batch size         1
            Entities           271
            Relations          2
            Shuffle            True
            Train triples      985
            Validation triples 24
            Test triples       24


        >>> for data in dataset:
        ...     print(data)
        ...     break
        {'sample': tensor([[ 69,   0, 269]]), 'weight': tensor([0.1601]), 'mode': 'head-batch'}

        >>> import torch

        >>> dataset = datasets.CountriesS3(batch_size=2, classification=False,
        ...     pre_compute=False, shuffle=True, seed=42)

        >>> for data in dataset:
        ...     assert data['sample'].shape == torch.Size([2, 3])
        ...     assert data['weight'].shape == torch.Size([2])
        ...     break

        >>> dataset = datasets.CountriesS3(batch_size=2, classification=True,
        ...     pre_compute=True, shuffle=True, seed=42)

        >>> for data in dataset:
        ...     assert data['sample'].shape == torch.Size([2, 2])
        ...     assert data['y'].shape == torch.Size([2, 271])
        ...     assert data['mode'] == 'classification'
        ...     break

        >>> dataset = datasets.CountriesS3(batch_size=2, classification=True,
        ...     pre_compute=False, shuffle=True, seed=42)

        >>> for data in dataset:
        ...     assert data['sample'].shape == torch.Size([2, 2])
        ...     assert data['y'].shape == torch.Size([2, 271])
        ...     assert data['mode'] == 'classification'
        ...     break

    References:
        1. [Bouchard, Guillaume, Sameer Singh, and Theo Trouillon. "On approximate reasoning capabilities of low-rank vector spaces." 2015 AAAI Spring Symposium Series. 2015.](https://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10257/10026)
        2. [Datasets for Knowledge Graph Completion with Textual Information about Entities](https://github.com/villmow/datasets_knowledge_embedding)

    """

    def __init__(self, batch_size, classification=False, shuffle=True, pre_compute=True,
                 num_workers=1, seed=None):

        self.filename = 'countries_s3'

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f'{path}/train.csv'),
            valid=read_csv(file_path=f'{path}/valid.csv'),
            test=read_csv(file_path=f'{path}/test.csv'),
            classification=classification,
            pre_compute=pre_compute,
            entities=read_json(f'{path}/entities.json'),
            relations=read_json(f'{path}/relations.json'),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed
        )
