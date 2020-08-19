import torch

import numpy as np

from ..utils import FetchToPredict
from ..utils import make_prediction


__all__ = ['find_treshold', 'accuracy']


def find_treshold(model, X, y, batch_size, num_workers=1, device='cuda'):
    """Computes best threshold and associated accuracy for triplet prediction task.

    Parameters:

        model (kdmkb.models): Model.
        X (list): Triplets.
        y (list): Label set to 1 if the triplet exists.
        batch_size (int): Size of the batch.
        num_workers (int): Number of worker to load input dataset.

    Example:

        >>> from kdmkb import evaluation
        >>> from kdmkb import datasets
        >>> from kdmkb import models

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Umls(batch_size=2)

        >>> model = models.TransE(
        ...     n_entity = dataset.n_entity,
        ...     n_relation = dataset.n_relation,
        ...     hidden_dim = 3,
        ...     gamma = 6
        ... )

        >>> evaluation.find_treshold(
        ...     model = model,
        ...     X = dataset.classification_valid['X'],
        ...     y = dataset.classification_valid['y'],
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        {'threshold': 1.924787, 'accuracy': 0.513803}

        >>> evaluation.accuracy(
        ...     model = model,
        ...     X = dataset.classification_valid['X'],
        ...     y = dataset.classification_valid['y'],
        ...     threshold = 1.924787,
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        0.513803

        >>> evaluation.accuracy(
        ...     model = model,
        ...     X = dataset.classification_test['X'],
        ...     y = dataset.classification_test['y'],
        ...     threshold = 1.924787,
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        0.499243

    """

    with torch.no_grad():

        y_pred = make_prediction(
            model=model,
            dataset=X,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        positive, negative = _get_positive_negative(
            y_pred=y_pred,
            y_true=y
        )

        return _compute_best_treshold(y_pred=y_pred.cpu().numpy(), positive=positive, negative=negative)


def accuracy(model, X, y, threshold, batch_size, num_workers=1, device='cuda'):
    """Find the threshold which maximize accuracy given inputs parameters.

    Parameters:

        model (kdmkb.models): Model.
        X (list): Triplets.
        y (list): Label set to 1 if the triplet exists.
        threshold (float): Treshold to classify a triplet as existing or not.
        batch_size (int): Size of the batch.
        num_workers (int): Number of worker to load input dataset.

    Example:

        >>> from kdmkb import evaluation
        >>> from kdmkb import datasets
        >>> from kdmkb import models

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Umls(batch_size=2)

        >>> model = models.TransE(
        ...     n_entity = dataset.n_entity,
        ...     n_relation = dataset.n_relation,
        ...     hidden_dim = 3,
        ...     gamma = 6
        ... )

        >>> evaluation.find_treshold(
        ...     model = model,
        ...     X = dataset.classification_valid['X'],
        ...     y = dataset.classification_valid['y'],
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        {'threshold': 1.924787, 'accuracy': 0.513803}

        >>> evaluation.accuracy(
        ...     model = model,
        ...     X = dataset.classification_valid['X'],
        ...     y = dataset.classification_valid['y'],
        ...     threshold = 1.924787,
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        0.513803

        >>> evaluation.accuracy(
        ...     model = model,
        ...     X = dataset.classification_test['X'],
        ...     y = dataset.classification_test['y'],
        ...     threshold = 1.924787,
        ...     batch_size = 10,
        ...     device = 'cpu',
        ... )
        0.499243


    """
    with torch.no_grad():

        y_pred = make_prediction(
            model=model,
            dataset=X,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        positive, negative = _get_positive_negative(
            y_pred=y_pred,
            y_true=y
        )

        return _accuracy(
            positive=positive,
            negative=negative,
            threshold=threshold
        )


def _accuracy(positive, negative, threshold):
    """Computes accuracy.

    Example:

        >>> import numpy as np

        >>> negative = np.array([1, 2, 3, 4, 5])
        >>> positive = np.array([5, 6, 7, 8, 9])

        >>> _accuracy(positive, negative, threshold = 5)
        0.9

    """
    tp = np.count_nonzero(positive >= threshold)
    tn = np.count_nonzero(negative < threshold)
    return (tp + tn) / (len(positive) + len(negative))


def _get_positive_negative(y_true, y_pred):
    """Splits the output scores of the model according to the target variable.

    Example:

        >>> y_pred = [1, 2, 3, 4, 5]
        >>> y_true = [-1, -1, -1, 1., 1, 1]

        >>> positive, negative = _get_positive_negative(y_true=y_true, y_pred=y_pred)

        >>> positive
        array([4, 5])

        >>> negative
        array([1, 2, 3])

    """
    positive = []
    negative = []

    for pred, label in zip(y_pred, y_true):
        if label == 1:
            positive.append(pred)
        else:
            negative.append(pred)

    return np.array(positive), np.array(negative)


def _compute_best_treshold(y_pred, positive, negative):
    """Find the best treshold for triplet classification. Every triplets with a score >= treshold
    can be considered as existing triplets.

    Example:

        >>> import numpy as np

        >>> y_pred = np.array([1, 2, 3, 4, 5])
        >>> positive = np.array([4, 5])
        >>> negative = np.array([1, 2, 3])

        >>> _compute_best_treshold(y_pred=y_pred, positive=positive, negative=negative)
        {'threshold': 4, 'accuracy': 1.0}

    """
    best_accuracy = 0
    best_treshold = None

    for threshold in y_pred:

        accuracy = _accuracy(
            positive=positive,
            negative=negative,
            threshold=threshold
        )

        if accuracy > best_accuracy:
            best_treshold = threshold
            best_accuracy = accuracy

    return {'threshold': best_treshold, 'accuracy': best_accuracy}
