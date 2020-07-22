from torch.utils import data
import torch

import pathlib

from .base import TrainDataset
from .base import TestDataset


__all__ = ['Fetch']


class Fetch:
    """Iter over a dataset.

    The Fetch class allows to iterate on the data of a dataset. Fetch takes entities as input,
    relations, training data and optional validation and test data. Training data, validation and
    testing must be organized in the form of a triplet list. Entities and relations must be in a
    dictionary where the key is the label of the entity or relationship and the value must be the
    index of the entity / relation.

    Parameters:
        train (list): Training set.
        valid (list): Validation set.
        test (list): Testing set.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        batch_size (int): Size of the batch.
        shuffle (bool): Whether to shuffle the dataset or not.
        num_workers (int): Number of workers dedicated to iterate on the dataset.
        seed (int): Random state.

    Attributes:
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.

    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    Example:

        >>> from kdmkb import datasets

        >>> entities = {
        ...    0: 'bicycle',
        ...    1: 'bike',
        ...    2: 'car',
        ...    3: 'truck',
        ...    4: 'automobile',
        ...    5: 'brand',
        ... }

        >>> relations = {
        ...     0: 'is_a',
        ...     1: 'not_a',
        ... }

        >>> train = [
        ...     (0, 0, 1),
        ...     (1, 1, 2),
        ...     (2, 0, 4),
        ... ]

        >>> test = [
        ...    (3, 1, 2),
        ...    (4, 1, 5),
        ... ]

        >>> dataset = datasets.Fetch(train=train, test=test, entities=entities, relations=relations,
        ...     batch_size=1, seed=42)

        >>> dataset
        Fetch dataset
            Batch size          1
            Entities            6
            Relations           2
            Shuffle             False
            Train triples       3
            Validation triples  0
            Test triples        2

        # Iterate over the first three samples of the input training set:
        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(dataset)
        ...     print(positive_sample, weight, mode)
        tensor([[0, 0, 1]]) tensor([0.3536]) tail-batch
        tensor([[0, 0, 1]]) tensor([0.3536]) head-batch
        tensor([[1, 1, 2]]) tensor([0.3536]) tail-batch

    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(
            self, train, entities, relations, batch_size, valid=[], test=[], shuffle=False,
            num_workers=1, seed=None):
        self.train = train
        self.valid = valid
        self.test = test
        self.entities = entities
        self.relations = relations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed
        self.step = 0

        self.n_entity = len(entities)
        self.n_relation = len(relations)

        self.fetch_head = self.fetch(self.get_train_loader(mode='head-batch'))
        self.fetch_tail = self.fetch(self.get_train_loader(mode='tail-batch'))

        if self.seed:
            torch.manual_seed(self.seed)

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.fetch_head)
        else:
            data = next(self.fetch_tail)
        return data

    @property
    def _repr_title(self):
        return f'{self.__class__.__name__} dataset'

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.
        This property can be overriden in order to modify the output of the __repr__ method.
        """

        return {
            'Batch size': f'{self.batch_size}',
            'Entities': f'{self.n_entity}',
            'Relations': f'{self.n_relation}',
            'Shuffle': f'{self.shuffle}',
            'Train triples': f'{len(self.train)}',
            'Validation triples': f'{len(self.valid)}',
            'Test triples': f'{len(self.test)}'
        }

    def __repr__(self):

        l_len = max(map(len, self._repr_content.keys()))
        r_len = max(map(len, self._repr_content.values()))

        return (
            f'{self._repr_title}\n' +
            '\n'.join(
                k.rjust(l_len) + '  ' + v.ljust(r_len)
                for k, v in self._repr_content.items()
            )
        )

    def test_dataset(self, batch_size):
        return self.test_stream(triples=self.test, batch_size=batch_size)

    def validation_dataset(self, batch_size):
        return self.test_stream(triples=self.valid, batch_size=batch_size)

    @ staticmethod
    def fetch(dataloader):
        while True:
            yield from dataloader

    def test_stream(self, triples, batch_size):
        head_loader = self._get_test_loader(
            triples=triples, batch_size=batch_size, mode='head-batch')

        tail_loader = self._get_test_loader(
            triples=triples, batch_size=batch_size, mode='tail-batch')

        return [head_loader, tail_loader]

    def get_train_loader(self, mode):
        dataset = TrainDataset(
            triples=self.train, entities=self.entities, relations=self.relations, mode=mode,
            seed=self.seed)

        return data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, collate_fn=TrainDataset.collate_fn)

    def _get_test_loader(self, triples, batch_size, mode):
        test_dataset = TestDataset(
            triples=triples, true_triples=self.train + self.test + self.valid,
            entities=self.entities, relations=self.relations, mode=mode)

        return data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn)
