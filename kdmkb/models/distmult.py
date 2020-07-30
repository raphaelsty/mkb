import torch

from . import base

__all__ = ['DistMult']


class DistMult(base.BaseModel):
    """Dist-Mult model.

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

        >>> model = models.DistMult(hidden_dim = 3, n_entity = 2, n_relation = 2, gamma = 1)

        >>> model
        DistMult model
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
        1. [Yang, Bishan, et al. "Embedding entities and relations for learning and inference in knowledge bases." arXiv preprint arXiv:1412.6575 (2014).](https://arxiv.org/pdf/1412.6575.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim,
                         n_entity=n_entity, n_relation=n_relation, gamma=gamma)

    def forward(self, sample):
        head, relation, tail, shape = self.batch(sample=sample)
        score = (head * relation) * tail
        score = score.sum(dim=2)
        return score.view(shape)
