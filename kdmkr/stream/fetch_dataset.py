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

            >>> train = [
            ...     (1, 1, 2),
            ...     (2, 2, 3),
            ... ]

            >>> test = [
            ...    (1, 2, 2),
            ...    (4, 3, 5),
            ... ]

            >>> dataset = stream.FetchDataset(train=train, test=test, negative_sample_size=1,
            ...    batch_size=1, seed=42)

            # Iterate over the first three samples of the input training set:
            >>> for _ in range(3):
            ...     positive_sample, negative_sample, weight, mode = next(dataset)
            ...     print(positive_sample, negative_sample, weight, mode)
            tensor([[1, 1, 2]]) tensor([[3]]) tensor([0.3536]) tail-batch
            tensor([[1, 1, 2]]) tensor([[3]]) tensor([0.3536]) head-batch
            tensor([[2, 2, 3]]) tensor([[2]]) tensor([0.3536]) tail-batch

            >>> assert dataset.n_entity == 5

            >>> assert dataset.n_relation == 3

    """
    def __init__(self, train, negative_sample_size, valid=[], test=[], batch_size=1,
        shuffle=False, num_workers=1, seed=None):
        self.train = train
        self.valid = valid
        self.test = test
        self.negative_sample_size = negative_sample_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed
        self.step = 0

        # Number of entities across train, valid and test dataset:
        self.n_entity = len(set(itertools.chain.from_iterable([
            [head, tail] for head, _, tail in itertools.chain(train, valid, test)])))

        # Number of relations across train, valid and test dataset:
        self.n_relation = len(set(
            [relation for _, relation, _ in itertools.chain(train, valid, test)]))

        self.fetch_head = self.fetch(self.get_train_loader(mode='head-batch'))
        self.fetch_tail = self.fetch(self.get_train_loader(mode='tail-batch'))

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.fetch_head)
        else:
            data = next(self.fetch_tail)
        return data

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
        dataset = TrainDataset(triples=self.train, n_entity=self.n_entity,
            n_relation=self.n_relation, negative_sample_size=self.negative_sample_size, mode=mode,
            seed=self.seed)

        return data.DataLoader(dataset=dataset, batch_size=self.batch_size,
            shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=TrainDataset.collate_fn)

    def test_stream(self, triples, batch_size):
        head_loader = self._get_test_loader(triples=triples, batch_size=batch_size,
            mode='head-batch')
        tail_loader = self._get_test_loader(triples=triples, batch_size=batch_size,
            mode='tail-batch')
        return [head_loader, tail_loader]

    def _get_test_loader(self, triples, batch_size, mode):
        test_dataset = TestDataset(triples=triples,
            all_true_triples=self.train + self.test + self.valid, n_entity = self.n_entity,
            n_relation=self.n_relation, mode=mode)
        return data.DataLoader(dataset=test_dataset, batch_size=batch_size,
            num_workers=self.num_workers, collate_fn=TestDataset.collate_fn)
