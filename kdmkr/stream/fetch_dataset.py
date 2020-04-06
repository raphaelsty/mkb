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

            >>> triples = [
            ...     (1, 1, 2),
            ...     (2, 2, 3),
            ... ]

            >>> n_entity = 3
            >>> n_relation = 2

            >>> dataset = stream.FetchDataset(triples=triples, n_entity=n_entity,
            ...     n_relation=n_relation, batch_size=1, negative_sample_size=1, seed=42)

            >>> for _ in range(3):
            ...     positive_sample, negative_sample, weight, mode = next(dataset)
            ...     print(positive_sample, negative_sample, weight, mode)
            tensor([[1, 1, 2]]) tensor([[0]]) tensor([0.3536]) tail-batch
            tensor([[1, 1, 2]]) tensor([[2]]) tensor([0.3536]) head-batch
            tensor([[2, 2, 3]]) tensor([[2]]) tensor([0.3536]) tail-batch

    """
    def __init__(self, triples, negative_sample_size, batch_size=1, shuffle=False, num_workers=1,
        seed=None):

        # TODO update: do not take account of entities and relations in test and validation set.
        n_entity = len(set(itertools.chain.from_iterable(
            [[head, tail] for head, _, tail in triples])))

        n_relation = len(set([relation for _, relation, _ in triples]))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.head_dataset = TrainDataset(triples=triples, n_entity=n_entity, n_relation=n_relation,
            negative_sample_size=negative_sample_size, mode='head-batch', seed=seed)

        self.tail_dataset = TrainDataset(triples=triples, n_entity=n_entity, n_relation=n_relation,
            negative_sample_size=negative_sample_size, mode='tail-batch', seed=seed)

        self.head_loader = data.DataLoader(dataset=self.head_dataset,
            batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
            collate_fn=self.head_dataset.collate_fn)

        self.tail_loader = data.DataLoader(dataset=self.tail_dataset,
            batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
            collate_fn=self.tail_dataset.collate_fn)

        self.step = 0

        self.fetch_head = self.fetch(self.head_loader)
        self.fetch_tail = self.fetch(self.tail_loader)

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.fetch_head)
        else:
            data = next(self.fetch_tail)
        return data

    @staticmethod
    def fetch(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            yield from dataloader
