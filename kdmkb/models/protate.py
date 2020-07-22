import torch

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

        >>> from kdmkb import models

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> model = models.pRotatE(hidden_dim = 3, n_entity = 2, n_relation = 2, gamma = 1)

        >>> model
        pRotatE model
            Entities embeddings dim  3
            Relations embeddings dim 3
            Gamma                    1.0
            Number of entities       2
            Number of relations      2

        >>> model.embeddings['entities']
        {0: tensor([ 0.7645,  0.8300, -0.2343]), 1: tensor([ 0.9186, -0.2191,  0.2018])}

        >>> model.embeddings['relations']
        {0: tensor([-0.4869,  0.5873,  0.8815]), 1: tensor([-0.7336,  0.8692,  0.1872])}


    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim,
                         n_entity=n_entity, n_relation=n_relation, gamma=gamma)
        self.pi = 3.14159262358979323846

    def forward(self, sample, mode='default'):
        head, relation, tail = self.head_relation_tail(
            sample=sample, mode=mode)
        phase_head = head/(self.embedding_range.item()/self.pi)
        phase_relation = relation/(self.embedding_range.item()/self.pi)
        phase_tail = tail/(self.embedding_range.item()/self.pi)
        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail
        score = torch.sin(score)
        score = torch.abs(score)
        return self.gamma.item() - score.sum(dim=2) * self.modulus

    def distill(self, sample):
        """Distillation method of ProtatE."""
        head, relation, tail = self.distillation_batch(sample)
        phase_head = head/(self.embedding_range.item()/self.pi)
        phase_relation = relation/(self.embedding_range.item()/self.pi)
        phase_tail = tail/(self.embedding_range.item()/self.pi)
        score = phase_head + (phase_relation - phase_tail)
        score = torch.sin(score)
        score = torch.abs(score)
        return self.gamma.item() - score.sum(dim=-1) * self.modulus
