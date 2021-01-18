from ..utils import FetchToPredict
from ..utils import make_prediction

import torch

import numpy as np

import tqdm


__all__ = ['find_threshold', 'accuracy']


def accuracy(model, X, y, threshold, batch_size, num_workers=1, device='cuda'):
    """Find the threshold which maximize accuracy given inputs parameters.

    Parameters:

        model (mkb.models): Model.
        X (list): Triplets.
        y (list): Label set to 1 if the triplet exists.
        threshold (float): threshold to classify a triplet as existing or not.
        batch_size (int): Size of the batch.
        num_workers (int): Number of worker to load input dataset.

    Example:

        >>> from mkb import evaluation
        >>> from mkb import datasets
        >>> from mkb import models

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Umls(batch_size=2)

        >>> model = models.TransE(
        ...     entities = dataset.entities,
        ...     relations = dataset.relations,
        ...     hidden_dim = 3,
        ...     gamma = 6
        ... )

        >>> evaluation.find_threshold(
        ...     model = model,
        ...     X = dataset.classification_valid['X'],
        ...     y = dataset.classification_valid['y'],
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        1.9384804

        >>> evaluation.accuracy(
        ...     model = model,
        ...     X = dataset.classification_valid['X'],
        ...     y = dataset.classification_valid['y'],
        ...     threshold = 1.9384804,
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        0.5130368098159509

        >>> evaluation.accuracy(
        ...     model = model,
        ...     X = dataset.classification_test['X'],
        ...     y = dataset.classification_test['y'],
        ...     threshold = 1.9384804,
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        0.49924357034795763


    """
    with torch.no_grad():

        y_pred = make_prediction(
            model=model,
            dataset=X,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        y_pred = y_pred.cpu().numpy()

        return _accuracy(
            y_true=y,
            y_pred=y_pred,
            threshold=threshold,
        )


def find_threshold(model, X, y, batch_size, num_workers=1, device='cuda'):
    """Find the best treshold according to input model and dataset.

    >>> from sklearn import metrics

    >>> y_true = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
    >>> y_pred = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]

    >>> false_positive_rates, true_positive_rates, tresholds = metrics.roc_curve(
    ...    y_true=y_true,
    ...    y_score=y_pred
    ... )

    >>> tresholds[np.argmax(true_positive_rates - false_positive_rates)]
    6

    """
    from sklearn import metrics

    with torch.no_grad():

        y_pred = make_prediction(
            model=model,
            dataset=X,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        y_pred = y_pred.cpu().numpy()

    false_positive_rates, true_positive_rates, tresholds = metrics.roc_curve(
        y_true=y,
        y_score=y_pred
    )

    return tresholds[np.argmax(true_positive_rates - false_positive_rates)]


def _accuracy(y_pred, y_true, threshold):
    """Find the best threshold for triplet classification. Every triplets with a score >= threshold
    can be considered as existing triplets.

    Example:

        #>>> from mkb import evaluation
        >>> import numpy as np

        >>> y_true = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
        >>> y_pred = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])

        >>> _accuracy(y_pred = y_pred, y_true = y_true, threshold = 5)
        0.9

    """
    accuracy = 0

    for i in range(len(y_pred)):

        if y_pred[i] >= threshold and y_true[i] > 0:

            accuracy += 1

        if y_pred[i] < threshold and y_true[i] <= 0:

            accuracy += 1

    return accuracy / len(y_pred)
