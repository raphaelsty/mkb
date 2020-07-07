# Reference: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
import torch

from . import base

__all__ = ['DistMult']

class DistMult(base.BaseModel):
    """DistMult

    Example:

        >>> from kdmkr import models

        >>> model = models.DistMult(hidden_dim = 10, n_entity = 2, n_relation = 2, gamma = 1)

        >>> model
        DistMult({'entity_dim': 10, 'relation_dim': 10, 'gamma': 1.0})

    """
    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim,
            n_entity=n_entity, n_relation=n_relation, gamma=gamma)

    def forward(self, sample, mode='default'):
        head, relation, tail = self.head_relation_tail(sample=sample, mode=mode)
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail
        return score.sum(dim=2)

    def distill(self, sample):
        head, relation, tail = self.distillation_batch(sample)
        score = head * (relation * tail)
        return score.sum(dim=-1)