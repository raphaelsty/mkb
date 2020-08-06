
import torch

from . import FetchToPredict

__all__ = ['make_prediction']


def make_prediction(model, dataset, batch_size, num_workers=1):
    """Compute predicion for given model and tripets.

    Parameters:
        model (kdmkb.models): Model.
        dataset (list): List of triplets.
        batch_size (int): Size of the batch.
        num_workers (int): Number of workers to load the input dataset.

    Example:

        >>> from kdmkb import utils
        >>> from kdmkb import models
        >>> from kdmkb import datasets

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Umls(batch_size=2)

        >>> model = models.TransE(
        ...     n_entity = dataset.n_entity,
        ...     n_relation = dataset.n_relation,
        ...     hidden_dim = 3,
        ...     gamma = 6
        ... )

        >>> utils.make_prediction(
        ...     model = model,
        ...     dataset = dataset.test[:3], # Only compute scores for 3 samples of the test set.
        ...     batch_size = 20,
        ... )
        tensor([-2.4270, -2.1356, -2.4053])

    """
    with torch.no_grad():

        y_pred = []

        for x in FetchToPredict(dataset=dataset, batch_size=batch_size, num_workers=num_workers):
            y_pred.append(model(x))

        return torch.cat(y_pred).flatten()
