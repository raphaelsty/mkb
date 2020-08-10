import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Adversarial']


class Adversarial(nn.Module):
    """Self-adversarial negative sampling loss function.

    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __call__(self, positive_score, negative_score, weight):
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        negative_score = (
            F.softmax(negative_score * self.alpha, dim=1).detach() *
            F.logsigmoid(-negative_score)
        ).sum(dim=1)

        positive_loss = - (weight * positive_score).sum()/weight.sum()
        negative_loss = - (weight * negative_score).sum()/weight.sum()
        return (positive_loss + negative_loss) / 2
