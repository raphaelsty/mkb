import itertools

from torch.utils import data

from .base import TrainDataset
from .base import TestDataset


__all__ = ['FetchDataset']


class FetchDataset:
    """Fetch Dataset

    Example:

        :
            >>> from kdmkr import stream

            >>> entities = {
            ...    0: 'bicycle',
            ...    1: 'velo',
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
            ...     (0, 0, 2),
            ...     (1, 0, 2),
            ...     (2, 1, 3),
            ... ]

            >>> test = [
            ...    (3, 1, 2),
            ...    (4, 1, 5),
            ... ]

            >>> dataset = stream.FetchDataset(train=train, test=test, entities=entities,
            ...    relations=relations, batch_size=1, seed=42)

            >>> dataset
            FetchDataset({'batch_size': 1})

            # Iterate over the first three samples of the input training set:
            >>> for _ in range(3):
            ...     positive_sample, weight, mode = next(dataset)
            ...     print(positive_sample, weight, mode)
            tensor([[0, 0, 2]]) tensor([0.3333]) tail-batch
            tensor([[0, 0, 2]]) tensor([0.3333]) head-batch
            tensor([[1, 0, 2]]) tensor([0.3333]) tail-batch

    """
    def __init__(self, train, entities, relations, valid=[], test=[],
        batch_size=1, shuffle=False, num_workers=1, seed=None):
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

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.fetch_head)
        else:
            data = next(self.fetch_tail)
        return data


    @property
    def get_params(self):
        return {
            'batch_size': self.batch_size,
        }


    def __repr__(self):
        return f'{self.__class__.__name__}({self.get_params})'


    def __str__(self):
        return self

    def test_dataset(self, batch_size):
        return self.test_stream(triples=self.test, batch_size=batch_size)

    def validation_dataset(self, batch_size):
        return self.test_stream(triples=self.valid, batch_size=batch_size)

    @staticmethod
    def fetch(dataloader):
        """
        Transform a PyTorch Dataloader into generator.
        """
        while True:
            yield from dataloader

    def get_train_loader(self, mode):
        dataset = TrainDataset(triples=self.train, entities=self.entities, relations=self.relations,
            mode=mode, seed=self.seed)

        return data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, collate_fn=TrainDataset.collate_fn)

    def test_stream(self, triples, batch_size):
        head_loader = self._get_test_loader(triples=triples, batch_size=batch_size,
            mode='head-batch')
        tail_loader = self._get_test_loader(triples=triples, batch_size=batch_size,
            mode='tail-batch')
        return [head_loader, tail_loader]

    def _get_test_loader(self, triples, batch_size, mode):
        test_dataset = TestDataset(triples=triples,
            all_true_triples=self.train + self.test + self.valid, entities=self.entities,
            relations=self.relations, mode=mode)
        return data.DataLoader(dataset=test_dataset, batch_size=batch_size,
            num_workers=self.num_workers, collate_fn=TestDataset.collate_fn)
