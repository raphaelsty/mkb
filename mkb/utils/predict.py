from torch.utils import data
from torch.utils.data import Dataset

import torch

__all__ = ['FetchToPredict', 'make_prediction']


class FetchToPredict(Dataset):
    """Fetch input dataset and format data to make prediction.

    Parameters:
        dataset (list): List of triplets.
        batch_size (int): Size of the output batch.
        num_workers (int): Number of workers to use to fetch input data.

    Example:

        >>> from mkb import utils

        >>> dataset = [
        ...    (0, 0, 1),
        ...    (1, 0, 2),
        ...    (1, 1, 3),
        ... ]

        >>> for x in FetchToPredict(dataset = dataset, batch_size = 2):
        ...     print(x)
        tensor([[0, 0, 1],
            [1, 0, 2]])
        tensor([[1, 1, 3]])

    """

    def __init__(self, dataset, batch_size, num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __getitem__(self, idx):
        return torch.LongTensor(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        yield from data.DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(data):
        """Reshape output data when calling train dataset loader."""
        return torch.stack(data, dim=0)


def make_prediction(model, dataset, batch_size, num_workers=1, device='cuda'):
    """Compute predicion for given model and tripets.

    Parameters:
        model (mkb.models): Model.
        dataset (list): List of triplets.
        batch_size (int): Size of the batch.
        num_workers (int): Number of workers to load the input dataset.

    Example:

        >>> from mkb import utils
        >>> from mkb import models
        >>> from mkb import datasets

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Umls(batch_size=2)

        >>> model = models.TransE(
        ...     entities = dataset.entities,
        ...     relations = dataset.relations,
        ...     hidden_dim = 3,
        ...     gamma = 6
        ... )

        >>> utils.make_prediction(
        ...     model = model,
        ...     dataset = dataset.test[:3], # Only compute scores for 3 samples of the test set.
        ...     batch_size = 20,
        ...     device = 'cpu',
        ... )
        tensor([-2.4270, -2.1356, -2.4053])

    """
    with torch.no_grad():

        y_pred = []

        for x in FetchToPredict(dataset=dataset, batch_size=batch_size, num_workers=num_workers):

            y_pred.append(model(x.to(device)))

        return torch.cat(y_pred).flatten()
