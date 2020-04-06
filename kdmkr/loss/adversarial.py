import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Adversarial']


class Adversarial(nn.Module):
    """Self-adversarial negative sampling loss function.

    Parameters:
        positive_score (torch.Tensor):

    Reference:
        RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
        http://arxiv.org/abs/1902.10197

    """
    def __init__(self):
        pass

    def __call__(self, positive_score, negative_score, weight, alpha = 1e-5):
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        positive_score = (F.softmax(positive_score * alpha, dim = 1).detach() *
            F.logsigmoid(-positive_score)).sum(dim = 1)

        positive_loss = - (weight * positive_score).sum()/weight.sum()
        negative_loss = - (weight * positive_score).sum()/weight.sum()
        return (positive_loss + negative_loss) / 2