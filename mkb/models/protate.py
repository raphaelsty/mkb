from math import pi

import torch
import torch.nn as nn

from . import base

__all__ = ['pRotatE']


class pRotatE(base.BaseModel):
    """pRotatE model.

    Parameters:
        hiddem_dim (int): Embedding size of relations and entities.
        n_entity (int): Number of entities to consider.
        n_relation (int): Number of relations to consider.
        gamma (float): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.


    Example:

        >>> from mkb import models
        >>> from mkb import datasets

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.CountriesS1(batch_size = 2)

        >>> model = models.pRotatE(
        ...    hidden_dim = 3,
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    gamma = 1
        ... )

        >>> model
        pRotatE model
            Entities embeddings dim  3
            Relations embeddings dim  3
            Gamma  1.0
            Number of entities  271
            Number of relations  2

        >>> model.embeddings['entities']['oceania']
        tensor([ 0.4845,  0.8654, -0.6108])

        >>> model.embeddings['relations']['locatedin']
        tensor([ 0.3845,  0.5489, -0.2268])


    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, hidden_dim, entities, relations, gamma):
        super().__init__(hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim,
                         entities=entities, relations=relations, gamma=gamma)

        self.pi = pi

        self.modulus = nn.Parameter(
            torch.Tensor([[0.5 * self.embedding_range.item()]])
        )

    def forward(self, sample, negative_sample=None, mode=None):
        head, relation, tail, shape = self.batch(
            sample=sample,
            negative_sample=negative_sample,
            mode=mode
        )

        phase_head = head/(self.embedding_range.item()/self.pi)
        phase_relation = relation/(self.embedding_range.item()/self.pi)
        phase_tail = tail/(self.embedding_range.item()/self.pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus

        return score.view(shape)
