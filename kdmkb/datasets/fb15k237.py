import os
import pathlib

from .fetch import Fetch

from ..utils import read_csv
from ..utils import read_csv_classification
from ..utils import read_json


__all__ = ['Fb15k237']


class Fb15k237(Fetch):
    """Fb15k237 dataset.

    Fb15k237 aim to iterate over the associated dataset. It provide positive samples, corresponding
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

        >>> fb15k237 = datasets.Fb15k237(batch_size=1, shuffle=True, pre_compute=False, seed=42)

        >>> fb15k237
        Fb15k237 dataset
            Batch size  1
            Entities  14541
            Relations  237
            Shuffle  True
            Train triples  272115
            Validation triples  17535
            Test triples  20466

        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(fb15k237)
        ...     print(positive_sample, weight, mode)
        tensor([[5222,   24, 1165]]) tensor([0.2887]) tail-batch
        tensor([[8615,   12, 2350]]) tensor([0.3333]) head-batch
        tensor([[  16,   15, 4726]]) tensor([0.0343]) tail-batch

        >>> assert len(fb15k237.classification_valid['X']) == len(fb15k237.classification_valid['y'])
        >>> assert len(fb15k237.classification_test['X']) == len(fb15k237.classification_test['y'])

        >>> assert len(fb15k237.classification_valid['X']) == len(fb15k237.valid) * 2
        >>> assert len(fb15k237.classification_test['X']) == len(fb15k237.test) * 2

    References:
        1. [Toutanova, Kristina, et al. "Representing text for joint embedding of text and knowledge bases." Proceedings of the 2015 conference on empirical methods in natural language processing. 2015.](https://www.aclweb.org/anthology/D15-1174.pdf)
        2. [Datasets for Knowledge Graph Completion with Textual Information about Entities](https://github.com/villmow/datasets_knowledge_embedding)

    """

    def __init__(self, batch_size, classification=False, shuffle=True, pre_compute=True,
                 num_workers=1, seed=None):

        self.filename = 'fb15k237'

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
