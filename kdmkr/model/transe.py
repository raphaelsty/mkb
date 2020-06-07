import torch

from . import base

__all__ = ['TransE']

class TransE(base.Teacher):
    """TransE"""

    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim,
            n_entity=n_entity, n_relation=n_relation, gamma=gamma)

    def forward(self, sample, mode='default'):
        head, relation, tail = self.head_relation_tail(sample=sample, mode=mode)
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        return self.gamma.item() - torch.norm(score, p=1, dim=2)

    def distill(self, sample):
        head, relation, tail = self.distillation_batch(sample)
        score = head + (relation - tail)
        return self.gamma.item() - torch.norm(score, p=1, dim=-1)

    def _top_k(self, sample):
        """Method dedicated to compute the top k entities and relations for a given triplet."""
        head, relation, tail = self.head_relation_tail(sample=sample, mode='default')
        embedding_head     = tail - relation
        embedding_relation = tail - head
        embedding_tail     = head + relation
        return embedding_head, embedding_relation, embedding_tail