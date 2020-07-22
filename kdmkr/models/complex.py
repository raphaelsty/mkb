import torch

from . import base

__all__ = ['ComplEx']


class ComplEx(base.BaseModel):
    """ComplEx model.

    Parameters:
        hiddem_dim (int): Embedding size of relations and entities.
        n_entity (int): Number of entities to consider.
        n_relation (int): Number of relations to consider.
        gamma (float): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.

    Example:

        >>> from kdmkr import models

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> model = models.ComplEx(hidden_dim = 3, n_entity = 2, n_relation = 2, gamma = 1)

        >>> model
        ComplEx model
            Entities embeddings dim  6
            Relations embeddings dim 6
            Gamma                    1.0
            Number of entities       2
            Number of relations      2

        >>> model.embeddings['entities']
        {0: tensor([ 0.7645,  0.8300, -0.2343,  0.9186, -0.2191,  0.2018]), 1: tensor([-0.4869,  0.5873,  0.8815, -0.7336,  0.8692,  0.1872])}

        >>> model.embeddings['relations']
        {0: tensor([ 0.7388,  0.1354,  0.4822, -0.1412,  0.7709,  0.1478]), 1: tensor([-0.4668,  0.2549, -0.4607, -0.1173, -0.4062,  0.6634])}


    References:
        1. [Trouillon, Théo, et al. "Complex embeddings for simple link prediction." International Conference on Machine Learning (ICML), 2016.](http://proceedings.mlr.press/v48/trouillon16.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(
            hidden_dim=hidden_dim, relation_dim=hidden_dim*2, entity_dim=hidden_dim*2,
            n_entity=n_entity, n_relation=n_relation, gamma=gamma)

    def forward(self, sample, mode='default'):
        head, relation, tail = self.head_relation_tail(
            sample=sample, mode=mode)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':

            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score

        else:

            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        return score.sum(dim=2)

    def distill(self, sample):
        """Distillation method of ComplEx."""
        head, relation, tail = self.distillation_batch(sample)
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score + im_head * im_score
        return score.sum(dim=-1)
