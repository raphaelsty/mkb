# Reference: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
import torch
import torch.nn as nn

from . import base

__all__ = ['RotatE']

class RotatE(base.Teacher):
    """RotatE"""

    def __init__(self, hidden_dim, n_entity, n_relation, gamma):
        super().__init__(hidden_dim=hidden_dim, relation_dim=hidden_dim, entity_dim=hidden_dim*2,
            n_entity=n_entity, n_relation=n_relation, gamma=gamma)
        self.pi = 3.14159265358979323846
        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

    def forward(self, sample, mode='default'):
        head, relation, tail = self.head_relation_tail(sample=sample, mode=mode)
        re_head, im_head = torch.chunk(head, 2, dim = 2)
        re_tail, im_tail = torch.chunk(tail, 2, dim = 2)

        phase_relation = relation/(self.embedding_range.item()/self.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        return self.gamma.item() - score.sum(dim = 2)

    def distill(self, sample):
        head, relation, tail = self.distillation_batch(sample)

        re_head, im_head = torch.chunk(head, 2, dim = -1)
        re_tail, im_tail = torch.chunk(tail, 2, dim = -1)

        phase_relation = relation/(self.embedding_range.item()/self.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        return self.gamma.item() - score.sum(dim = -1)
