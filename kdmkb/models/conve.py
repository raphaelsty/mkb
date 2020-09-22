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

        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.CountriesS1(3, shuffle = False)

        >>> model = models.ConvE(
        ...     hidden_dim = (4, 4),
        ...     n_entity = dataset.n_entity,
        ...     n_relation = dataset.n_relation,
        ... )

        >>> model = model.eval()

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

        >>> sample = torch.tensor([
        ...    [0, 0],
        ...    [1, 1],
        ...    [1, 1]
        ... ])

        >>> y_pred = model(sample)

        >>> assert y_pred.shape == torch.Size([3, 271])

        ConvE needs to be in evaluation mode to make predictions.
        >>> model = model.eval()

        >>> sample = torch.tensor([[0, 0]])

        >>> y_pred = model(sample)

        >>> assert y_pred.shape == torch.Size([1, 271])

        >>> y_pred[0][266]
        tensor(-0.0131, grad_fn=<SelectBackward>)

        >>> sample = torch.tensor([[0, 0], [1, 1], [1, 1]])

        >>> y_pred = model(sample)

        >>> y_pred[0][266]
        tensor(-0.0131, grad_fn=<SelectBackward>)

        >>> y_pred[1][56]
        tensor(-0.0137, grad_fn=<SelectBackward>)

        >>> y_pred[2][14]
        tensor(0.0042, grad_fn=<SelectBackward>)

        >>> sample = torch.tensor([
        ...    [  0,   0, 266],
        ...    [  1,   1,  56],
        ...    [  1,   1,  14]
        ... ])

        >>> negative_sample = torch.tensor([
        ...    [266, 266], [56, 56], [14, 14]
        ... ])

        >>> model(sample, negative_sample, mode = 'tail-batch')
        tensor([[-0.0131, -0.0131],
                [-0.0137, -0.0137],
                [ 0.0042,  0.0042]], grad_fn=<ViewBackward>)

        >>> negative_sample = torch.tensor([
        ...    [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]
        ... ])

        >>> model(sample, negative_sample, mode = 'head-batch')
        tensor([[-0.0131, -0.0131, -0.0131, -0.0131, -0.0131, -0.0131],
                [-0.0137, -0.0137, -0.0137, -0.0137, -0.0137, -0.0137],
                [ 0.0042,  0.0042,  0.0042,  0.0042,  0.0042,  0.0042]],
           grad_fn=<ViewBackward>)


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
        layer_dropout=0.3,
        chunk_size=250,
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

        self.chunk_size = chunk_size

        self.conv_e = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(p=self.embedding_dropout),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=self.channels),
            nn.ReLU(),
            nn.Dropout2d(p=self.feature_map_dropout),
            Flatten(),
            nn.Linear(
                in_features=self.flattened_size,
                out_features=self.hidden_dim
            ),
            nn.Dropout(p=self.layer_dropout),
            nn.BatchNorm1d(num_features=self.hidden_dim),
            nn.ReLU()
        )

        nn.init.xavier_normal_(self.entity_embedding.weight.data)
        nn.init.xavier_normal_(self.relation_embedding.weight.data)

    def forward(self, sample, negative_sample=None, mode='default'):
        # Classification mode, returns probability distribution(s) of tails given head(s) and
        # relation(s). It is the mode dedicated to optimize BCELoss.
        if len(sample.shape) == 2 and sample.shape[1] == 2 and negative_sample is None:
            mode = 'classification'

        head, relation, tail, shape = self.batch(
            sample=sample, negative_sample=negative_sample, mode=mode)

        if mode == 'head-batch':
            # ConvE is not designed to predict head from a given relation and tail.
            # We divide input batch into chunk of size 250 to avoid explode RAM.
            return self.head_batch_chunk(
                head=head,
                relation=relation,
                tail=tail,
                shape=shape,
                chunk_size=self.chunk_size,
            )

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

        if mode == 'classification':
            scores = scores.mm(self.entity_embedding.weight.t())
        else:
            scores = scores.unsqueeze(1).bmm(tail)

        return scores.view(shape)

    def head_batch_chunk(self, head, relation, tail, shape, chunk_size):
        """
        Process head-batch input data into chunk to reduce RAM usage. ConvE is not designed to find
        the head that likely complete the triple (?, relation, tail).
        """
        list_scores = []

        size = max(head.shape[0] // chunk_size, 1)

        batch_h = torch.chunk(head, size)
        batch_r = torch.chunk(relation, size)
        batch_t = torch.chunk(tail, size)

        for h, r, t in zip(batch_h, batch_r, batch_t):

            h = h.view(
                -1,
                self.hidden_dim_w,
                self.hidden_dim_h
            )

            r = r.view(
                -1,
                self.hidden_dim_w,
                self.hidden_dim_h
            )

            t = t.view(
                h.shape[0],
                self.hidden_dim_w * self.hidden_dim_h,
                1
            )

            input_nn = torch.cat([h, r], dim=1).unsqueeze(dim=1)

            scores = self.conv_e(input_nn)

            scores = scores.unsqueeze(1).bmm(t)

            list_scores.append(scores)

        return torch.cat(list_scores).view(shape)
