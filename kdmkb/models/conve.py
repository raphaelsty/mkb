import torch
import torch.nn as nn

from . import base

__all__ = ['ConvE']


class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        x = x.view(n, -1)
        return x


class ConvE(base.BaseConvE):
    """ConvE model.

    Examples:

        >>> from kdmkb import datasets
        >>> from kdmkb import models

        >>> import torch

        >>> _ = torch.manual_seed(0)

        >>> dataset = datasets.CountriesS1(3, shuffle = False)

        >>> model = models.ConvE(
        ...     hidden_dim = (4, 4),
        ...     n_entity = dataset.n_entity,
        ...     n_relation = dataset.n_relation,
        ... )

        >>> model
        ConvE model
        Entities embeddings dim  16
        Relations embeddings dim 16
        Number of entities       271
        Number of relations      2
        Channels                 32
        Kernel size              3
        Embeddings dropout       0.2
        Feature map dropout      0.2
        Layer dropout            0.3

        >>> sample, weights, mode = next(dataset)

        >>> sample
        tensor([[  0,   0, 266],
                [  1,   1,  56],
                [  1,   1,  14]])

        >>> sample = torch.tensor([[0, 0, 266]])

        >>> model = model.eval()

        For each input sample, ConvE computes probability distribution for each target.
        >>> y_pred = model(sample)

        >>> y_pred.shape
        torch.Size([1, 271])

        >>> y_pred[0][:10]
        tensor([ 0.0000, -0.0717,  0.0243, -0.0185, -0.0627, -0.0075, -0.1589, -0.1653,
            -0.0657, -0.0612], grad_fn=<SliceBackward>)

    References:
        1. [Convolutional 2D Knowledge Graph Embeddings, Dettmers, Tim and Pasquale, Minervini and Pontus, Stenetorp and Riedel, Sebastian](https://arxiv.org/abs/1707.01476)
        2. [Convolutional 2D Knowledge Graph Embeddings resources.](https://github.com/TimDettmers/ConvE)
        3. [Implementation of ConvE](https://github.com/magnusja/ConvE/blob/master/model.py)

    """

    def __init__(
        self,
        hidden_dim,
        n_entity,
        n_relation,
        channels=32,
        kernel_size=3,
        embedding_dropout=0.2,
        feature_map_dropout=0.2,
        layer_dropout=0.3
    ):

        super().__init__(
            n_entity=n_entity,
            n_relation=n_relation,
            hidden_dim_w=hidden_dim[0],
            hidden_dim_h=hidden_dim[1],
            channels=channels,
            kernel_size=kernel_size,
            embedding_dropout=embedding_dropout,
            feature_map_dropout=feature_map_dropout,
            layer_dropout=layer_dropout
        )

        self.conv_e = nn.Sequential(
            nn.Dropout(p=self.embedding_dropout),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.channels,
                kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.channels),
            nn.Dropout2d(p=self.feature_map_dropout),
            Flatten(),
            nn.Linear(
                in_features=self.flattened_size,
                out_features=self.hidden_dim
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.hidden_dim),
            nn.Dropout(p=self.layer_dropout)
        )

    def forward(self, sample):
        head = self.entity_embedding(sample[:, 0])
        relation = self.relation_embedding(sample[:, 1])

        head = head.view(
            -1,
            self.hidden_dim_w,
            self.hidden_dim_h
        )

        relation = relation.view(
            -1,
            self.hidden_dim_w,
            self.hidden_dim_h
        )

        input_nn = torch.cat([head, relation], dim=1).unsqueeze(dim=1)

        scores = self.conv_e(input_nn)

        return scores.mm(self.entity_embedding.weight.t())
