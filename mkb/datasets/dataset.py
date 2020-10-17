from torch.utils import data
import torch

from .base import TrainDataset
from .base import TestDataset

import copy


__all__ = ['Dataset']


class Dataset:
    """Iter over a dataset.

    The Dataset class allows to iterate on the data of a dataset. Dataset takes entities as input,
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
        classification_valid (dict[str, list]): Validation set dedicated to triplet classification
            task.
        classification_valid (dict[str, list]): Test set dedicated to triplet classification
            task.

    Attributes:
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.

    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    Example:

        >>> from mkb import datasets

        >>> train = [
        ...    ('🐝', 'is', 'animal'),
        ...    ('🐻', 'is', 'animal'),
        ...    ('🐍', 'is', 'animal'),
        ...    ('🦔', 'is', 'animal'),
        ...    ('🦓', 'is', 'animal'),
        ...    ('🦒', 'is', 'animal'),
        ...    ('🦘', 'is', 'animal'),
        ...    ('🦝', 'is', 'animal'),
        ...    ('🦞', 'is', 'animal'),
        ...    ('🦢', 'is', 'animal'),
        ... ]

        >>> test = [
        ...    ('🐝', 'is', 'animal'),
        ...    ('🐻', 'is', 'animal'),
        ...    ('🐍', 'is', 'animal'),
        ...    ('🦔', 'is', 'animal'),
        ...    ('🦓', 'is', 'animal'),
        ...    ('🦒', 'is', 'animal'),
        ...    ('🦘', 'is', 'animal'),
        ...    ('🦝', 'is', 'animal'),
        ...    ('🦞', 'is', 'animal'),
        ...    ('🦢', 'is', 'animal'),
        ... ]

        >>> dataset = datasets.Dataset(train=train, test=test, batch_size=2, seed=42)

        >>> dataset
        Dataset dataset
            Batch size  2
            Entities  11
            Relations  1
            Shuffle  True
            Train triples  10
            Validation triples  0
            Test triples  10

        >>> dataset.entities
        {'🐝': 0, '🐻': 1, '🐍': 2, '🦔': 3, '🦓': 4, '🦒': 5, '🦘': 6, '🦝': 7, '🦞': 8, '🦢': 9, 'animal': 10}


        >>> for data in dataset:
        ...     print(data)
        {'sample': tensor([[ 6,  0, 10],
            [ 5,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'head-batch'}
        {'sample': tensor([[ 4,  0, 10],
                [ 1,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'tail-batch'}
        {'sample': tensor([[ 4,  0, 10],
                [ 0,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'head-batch'}
        {'sample': tensor([[ 6,  0, 10],
                [ 9,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'tail-batch'}
        {'sample': tensor([[ 8,  0, 10],
                [ 9,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'head-batch'}
        {'sample': tensor([[ 5,  0, 10],
                [ 7,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'tail-batch'}
        {'sample': tensor([[ 2,  0, 10],
                [ 1,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'head-batch'}
        {'sample': tensor([[ 8,  0, 10],
                [ 2,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'tail-batch'}
        {'sample': tensor([[ 3,  0, 10],
                [ 7,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'head-batch'}
        {'sample': tensor([[ 0,  0, 10],
                [ 3,  0, 10]]), 'weight': tensor([0.2425, 0.2425]), 'mode': 'tail-batch'}

    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(
        self, train, batch_size, entities=None, relations=None, valid=None, test=None, shuffle=True,
        classification=False, pre_compute=True, num_workers=1, seed=None, classification_valid=None,
        classification_test=None
    ):
        self.train = train
        self.valid = valid
        self.test = test

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.classification = classification
        self.pre_compute = pre_compute
        self.num_workers = num_workers
        self.seed = seed

        # Construct mapping of entities and relations.
        # Each entity and relation as an id.
        if entities is None:
            self.entities = self.mapping_entities()
            self.train = [(self.entities[h], r, self.entities[t])
                          for h, r, t in self.train]

            if self.valid is not None:
                self.valid = [(self.entities[h], r, self.entities[t])
                              for h, r, t in self.valid]

            if self.test is not None:
                self.test = [(self.entities[h], r, self.entities[t])
                             for h, r, t in self.test]
        else:
            self.entities = entities

        if relations is None:

            self.relations = self.mapping_relations()

            self.train = [(h, self.relations[r], t) for h, r, t in self.train]

            if self.valid is not None:
                self.valid = [(h, self.relations[r], t)
                              for h, r, t in self.valid]

            if self.test is not None:
                self.test = [(h, self.relations[r], t)
                             for h, r, t in self.test]
        else:
            self.relations = relations

        # Number of distinct entities and relations.
        self.n_entity = len(self.entities)
        self.n_relation = len(self.relations)

        # The classification mode is dedicated to ConvE model.
        if self.classification:

            self.dataset = self.get_train_loader(mode='classification')
            self.len = int(len(self.dataset.dataset) / self.batch_size)

            # Needed for mkb to iterate over multiples datasets with single batch each time.
            # __next__ functionnality
            self.fetch_dataset = self.fetch(self.dataset)

        # When using TransE, RotatE, DistMult, ComplEx, pRotatE, classification mode must be set
        # to False.
        else:
            self.step = 0
            self.dataset_head = self.get_train_loader(mode='head-batch')
            self.dataset_tail = self.get_train_loader(mode='tail-batch')
            self.len = int((
                len(self.dataset_head.dataset) +
                len(self.dataset_tail.dataset)
            ) / self.batch_size)

            # Needed for mkb to iterate over multiples datasets with single batch each time.
            # __next__ functionnality
            self.step = 0
            self.fetch_head = self.fetch(self.dataset_head)
            self.fetch_tail = self.fetch(self.dataset_tail)

        # Dataset dedicated to triplet classification task. Optionnal.
        self.classification_valid = classification_valid
        self.classification_test = classification_test

        # Fix seed.
        if self.seed:
            torch.manual_seed(self.seed)

    def __iter__(self):
        if self.classification:
            yield from self.dataset
        else:
            for head, tail in zip(*[self.dataset_head, self.dataset_tail]):
                yield head
                yield tail

    def __next__(self):
        if self.classification:
            return next(self.fetch_dataset)
        else:
            self.step += 1
            if self.step % 2 == 0:
                return next(self.fetch_head)
            else:
                return next(self.fetch_tail)

    @staticmethod
    def fetch(dataloader):
        while True:
            yield from dataloader

    def __len__(self):
        return self.len

    @property
    def true_triples(self):
        """Get all true triples from the dataset."""
        true_triples = copy.deepcopy(self.train)

        if self.valid is not None:
            true_triples += self.valid

        if self.test is not None:
            true_triples += self.test

        return true_triples

    @property
    def train_triples(self):
        return self.train

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def _repr_title(self):
        return f'{self.name} dataset'

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
            'Train triples': f'{len(self.train) if self.train else 0}',
            'Validation triples': f'{len(self.valid) if self.valid else 0}',
            'Test triples': f'{len(self.test) if self.test else 0}'
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

    def test_stream(self, triples, batch_size):
        head_loader = self._get_test_loader(
            triples=triples, batch_size=batch_size, mode='head-batch')

        tail_loader = self._get_test_loader(
            triples=triples, batch_size=batch_size, mode='tail-batch')

        return [head_loader, tail_loader]

    def get_train_loader(self, mode):
        """Initialize train dataset loader."""
        dataset = TrainDataset(
            triples=self.train, entities=self.entities, relations=self.relations, mode=mode,
            pre_compute=self.pre_compute, seed=self.seed)

        if mode == 'classification':
            collate_fn = TrainDataset.collate_fn_classification
        else:
            collate_fn = TrainDataset.collate_fn

        return data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, collate_fn=collate_fn)

    def _get_test_loader(self, triples, batch_size, mode):
        """Initialize test dataset loader."""
        test_dataset = TestDataset(
            triples=triples, true_triples=self.train + self.test + self.valid,
            entities=self.entities, relations=self.relations, mode=mode)

        return data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn)

    def mapping_entities(self):
        """Construct mapping entities."""
        return {e: i for i, e in enumerate(
            dict.fromkeys(
                [h for h, _, _ in self.true_triples] + [t for _, _, t in self.true_triples]))}

    def mapping_relations(self):
        """Construct mapping relations."""
        return {r: i for i, r in enumerate(
            dict.fromkeys([r for _, r, _ in self.true_triples]))}
