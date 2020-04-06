import os
import zipfile

from ..stream import fetch_dataset
from ..stream import utils


__all__ = ['WN18RR']


class WN18RR(fetch_dataset.FetchDataset):
    """Iter over WN18RR

    Example:

        :
            >>> from kdmkr import datasets
            >>> import os

            >>> wn18rr = datasets.WN18RR(batch_size=1, negative_sample_size=1, seed=42)

            >>> file_path = f'{os.path.dirname(os.path.abspath(__file__))}/wn18rr.zip'
            >>> file = utils.open_filepath(file_path, compression='zip')

            >>> for row in file:
            ...     print(row)
            ...     break

    """
    def __init__(self, batch_size, negative_sample_size=1024, shuffle=False, num_workers=1,
        seed=None):

        # TODO: READ number of entities, relations across test, train and valid sets.

        super().__init__(triples=self.read_dataset('train.txt'), batch_size=batch_size,
            negative_sample_size=negative_sample_size, shuffle=shuffle, num_workers=num_workers,
            seed=seed)

    @property
    def n_entity(self):
        return self.n_entity

    @property
    def n_relation(self):
        return self.n_relation

    def read_dataset(self, dataset):
        file = zipfile.ZipFile('wn18rr.zip')
        dataset = file.open(f'wn18rr/{dataset}')
        dataset = dataset.readlines()
        dataset = [tuple(_.decode('utf-8').split('\n')[0].split('\t')) for _ in dataset]
        return dataset