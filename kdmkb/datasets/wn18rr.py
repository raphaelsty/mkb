import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_csv_classification
from ..utils import read_json

__all__ = ['Wn18rr']


class Wn18rr(Fetch):
    """Wn18rr dataset.

    Wn18rr aim to iterate over the associated dataset. It provide positive samples, corresponding
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

        >>> from kdmkb import datasets

        >>> wn18rr = datasets.Wn18rr(batch_size=1, shuffle=True, pre_compute=False, seed=42)

        >>> wn18rr
        Wn18rr dataset
            Batch size          1
            Entities            40943
            Relations           11
            Shuffle             True
            Train triples       86835
            Validation triples  3034
            Test triples        3134


        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(wn18rr)
        ...     print(positive_sample, weight, mode)
        tensor([[12241,     4, 33028]]) tensor([0.3333]) tail-batch
        tensor([[2635,    1, 6885]]) tensor([0.2500]) head-batch
        tensor([[ 1479,     0, 32588]]) tensor([0.3333]) tail-batch


        >>> assert len(wn18rr.classification_valid['X']) == len(wn18rr.classification_valid['y'])
        >>> assert len(wn18rr.classification_test['X']) == len(wn18rr.classification_test['y'])

        >>> assert len(wn18rr.classification_valid['X']) == len(wn18rr.valid) * 2
        >>> assert len(wn18rr.classification_test['X']) == len(wn18rr.test) * 2

    References:
        1. [Dettmers, Tim, et al. "Convolutional 2d knowledge graph embeddings." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.](https://arxiv.org/pdf/1707.01476.pdf)

    """

    def __init__(self, batch_size, classification=False, shuffle=True, pre_compute=True,
                 num_workers=1, seed=None):

        self.filename = 'wn18rr'

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f'{path}/train.csv'),
            valid=read_csv(file_path=f'{path}/valid.csv'),
            test=read_csv(file_path=f'{path}/test.csv'),
            entities=read_json(f'{path}/entities.json'),
            relations=read_json(f'{path}/relations.json'),
            batch_size=batch_size, shuffle=shuffle, classification=classification,
            pre_compute=pre_compute,
            num_workers=num_workers, seed=seed,
            classification_valid=read_csv_classification(
                f'{path}/classification_valid.csv'),
            classification_test=read_csv_classification(
                f'{path}/classification_test.csv'),
        )
