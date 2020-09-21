import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base

__all__ = ['TransH']


class TransH(base.BaseModel):
    """TransH model.

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

        >>> model = models.TransH(hidden_dim = 3, n_entity = 2, n_relation = 2, gamma = 1)

        >>> model
        TransH model
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
        1. [Wang, Z.; Zhang, J.; Feng, J.; and Chen, Z. 2014. Knowledge graph embedding by translating on hyperplanes. In Proceedings of AAAI, 1112–1119.](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531/8546)
        2. [An Open-source Framework for Knowledge Embedding implemented with PyTorch.](https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransH.py)

    """

    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim,
                         n_entity=n_entity, n_relation=n_relation, gamma=gamma)

        self.relation_norm = nn.Embedding(
            self.n_relation, self.relation_dim)

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        if e.shape != norm.shape:
            norm_up = torch.cat(
                [norm for _ in range(e.shape[1])], dim=-2).view(e.transpose(1, 2).shape)
            return (e - torch.sum(e.bmm(norm_up), -1, True)).bmm(norm.unsqueeze(1).transpose(1, 2))
        return e - torch.sum(e * norm, -1, True) * norm

    def forward(self, sample, negative_sample=None, mode=None):
        h, r, t, shape = self.batch(
            sample=sample,
            negative_sample=negative_sample,
            mode=mode
        )

        h = h.squeeze(1)
        r = r.squeeze(1)
        t = t.squeeze(1)

        if len(sample.shape) == 3:
            sample = sample.unsqueeze(1)

        norm = self.relation_norm(sample[:, 1])

        h = self._transfer(e=h, norm=norm)
        t = self._transfer(e=t, norm=norm)

        if mode == 'head-batch':
            score = h + (r - t)
        else:
            return (h + r).shape, t.shape
            score = (h + r) - t

        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)

        return score.view(shape)
