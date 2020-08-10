import torch
import torch.nn as nn

from . import base

__all__ = ['RotatE']


class RotatE(base.BaseModel):
    """RotatE

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

        >>> model = models.RotatE(hidden_dim = 3, n_entity = 2, n_relation = 2, gamma = 1)

        >>> model
        RotatE model
            Entities embeddings dim  6
            Relations embeddings dim 3
            Gamma                    1.0
            Number of entities       2
            Number of relations      2

        >>> model.embeddings['entities']
        {0: tensor([ 0.7645,  0.8300, -0.2343,  0.9186, -0.2191,  0.2018]),
                   1: tensor([-0.4869,  0.5873,  0.8815, -0.7336,  0.8692,  0.1872])}

        >>> model.embeddings['relations']
        {0: tensor([0.7388, 0.1354, 0.4822]),
                   1: tensor([-0.1412,  0.7709,  0.1478])}


    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(
            hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim*2,
            n_entity=n_entity, n_relation=n_relation, gamma=gamma)

        self.pi = 3.14159265358979323846
        self.modulus = nn.Parameter(torch.Tensor(
            [[0.5 * self.embedding_range.item()]]))

    def forward(self, sample):
        head, relation, tail, shape = self.batch(sample=sample)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation/(self.embedding_range.item() / self.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)
        return score.view(shape)
