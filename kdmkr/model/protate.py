import torch

from . import base

__all__ = ['pRotatE']

class pRotatE(base.Teacher):
    """pRotatE"""

    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim,
            n_entity=n_entity, n_relation=n_relation, gamma=gamma)
        self.pi = 3.14159262358979323846

    def forward(self, sample, mode='default'):
        head, relation, tail = self.head_relation_tail(sample=sample, mode=mode)
        phase_head = head/(self.embedding_range.item()/self.pi)
        phase_relation = relation/(self.embedding_range.item()/self.pi)
        phase_tail = tail/(self.embedding_range.item()/self.pi)
        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail
        score = torch.sin(score)
        score = torch.abs(score)
        return self.gamma.item() - score.sum(dim = 2) * self.modulus

    def distill(self, sample):
        head, relation, tail = self.distillation_batch(sample)
        phase_head = head/(self.embedding_range.item()/self.pi)
        phase_relation = relation/(self.embedding_range.item()/self.pi)
        phase_tail = tail/(self.embedding_range.item()/self.pi)
        score = phase_head + (phase_relation - phase_tail)
        score = torch.sin(score)
        score = torch.abs(score)
        return self.gamma.item() - score.sum(dim = -1) * self.modulus
