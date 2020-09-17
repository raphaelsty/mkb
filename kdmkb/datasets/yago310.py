import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_csv_classification
from ..utils import read_json


__all__ = ['Yago310']


class Yago310(Fetch):
    """Yago310 dataset.

    Yago310 aim to iterate over the associated dataset. It provide positive samples, corresponding
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

        >>> yago310 = datasets.Yago310(batch_size=1, shuffle=True, seed=42)

        >>> yago310
        Yago310 dataset
            Batch size         1
            Entities           123182
            Relations          37
            Shuffle            True
            Train triples      1079040
            Validation triples 5000
            Test triples       5000

        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(yago310)
        ...     print(positive_sample, weight, mode)
        tensor([[  7886,      0, 115154]]) tensor([0.1302]) tail-batch
        tensor([[  4032,      2, 116067]]) tensor([0.1768]) head-batch
        tensor([[ 68567,      1, 117352]]) tensor([0.1690]) tail-batch

        >>> assert len(yago310.classification_valid['X']) == len(yago310.classification_valid['y'])
        >>> assert len(yago310.classification_test['X']) == len(yago310.classification_test['y'])

        >>> assert len(yago310.classification_valid['X']) == len(yago310.valid) * 2
        >>> assert len(yago310.classification_test['X']) == len(yago310.test) * 2


    References:
        1. [Fabian M. Suchanek and Gjergji Kasneci and Gerhard Weikum, Yago: A Core of Semantic Knowledge, 16th International Conference on the World Wide Web, 2007](https://github.com/yago-naga/yago3)

    """

    def __init__(self, batch_size, classification=False, shuffle=True, pre_compute=False,
                 num_workers=1, seed=None):

        self.filename = 'yago310'

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
