import os

from ..stream import fetch_dataset
from ..utils import read_csv
from ..utils import read_json

__all__ = ['KDMKRFB15K237']


class KDMKRFB15K237(fetch_dataset.FetchDataset):
    """Iter over KDMKRFB15K237

    Example:

        :
            >>> from kdmkr import datasets
            >>> import torch

            >>> kdmkr_fb15k237 = datasets.KDMKRFB15K237(batch_size=1, negative_sample_size=1, shuffle=True,
            ... seed=42)

            >>> _ = torch.manual_seed(42)

            >>> for _ in range(3):
            ...     positive_sample, negative_sample, weight, mode = next(kdmkr_fb15k237)
            ...     print(positive_sample, negative_sample, weight, mode)
            tensor([[5196,   24, 1164]]) tensor([[7270]]) tensor([0.2887]) tail-batch
            tensor([[8539,   12, 2343]]) tensor([[7270]]) tensor([0.3333]) head-batch
            tensor([[  16,  240, 4709]]) tensor([[5390]]) tensor([0.0343]) tail-batch


    """
    def __init__(self, batch_size, negative_sample_size=1024, shuffle=False, num_workers=1,
        seed=None):

        self.folder    = 'kdmkr_fb15k237'
        self.directory = f'{os.path.dirname(os.path.realpath(__file__))}/{self.folder}'

        self.train_file_path = f'{self.directory}/train.csv'
        self.valid_file_path = f'{self.directory}/valid.csv'
        self.test_file_path  = f'{self.directory}/test.csv'

        self.entities_file_path  = f'{self.directory}/entities.json'
        self.relations_file_path = f'{self.directory}/relations.json'

        super().__init__(train=read_csv(file_path=self.train_file_path),
            valid=read_csv(file_path=self.valid_file_path), test=read_csv(file_path=self.test_file_path),
            batch_size=batch_size, negative_sample_size=negative_sample_size, shuffle=shuffle,
            num_workers=num_workers, seed=seed,
            entities=read_json(self.entities_file_path),
            relations=read_json(self.relations_file_path),
        )

