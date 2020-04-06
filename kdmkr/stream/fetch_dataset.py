from torch.utils import data

from .base import TrainDataset
from .base import TestDataset


__all__ = ['FetchDataset']


class FetchDataset(object):
    """Fetch Dataset """
    def __init__(self, triples, n_entity, n_relation, batch_size, negative_sample_size=1024,
        shuffle=False, num_workers=1):

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.head_dataset = TrainDataset(triples=triples, n_entity=n_entity, n_relation=n_relation,
            negative_sample_size=negative_sample_size, mode='head-batch')

        self.tail_dataset = TrainDataset(triples=triples, n_entity=n_entity, n_relation=n_relation,
            negative_sample_size=negative_sample_size, mode='head-batch')

        self.head_loader = data.DataLoader(dataset=self.head_dataset, batch_size=self.batch_size,
            shuffle=self.shuffle, num_workers=self.num_workers,
            collate_fn=self.head_dataset.collate_fn)

        self.tail_loader = data.DataLoader(dataset=self.tail_dataset, batch_size=self.batch_size,
            shuffle=self.shuffle, num_workers=self.num_workers,
            collate_fn=self.tail_dataset.collate_fn)

        self.fetch_head = self.fetch(self.head_loader)
        self.fetch_tail = self.fetch(self.tail_loader)

        self.step = 0

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
