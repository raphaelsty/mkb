import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KlDivergence"]


class KlDivergence(nn.Module):
    """Kullback-Leibler divergence loss dedicated to distillation.

    Inputs scores must have shape 3: (n distributions, n triplets per distribution, 3).

    References
    ----------
    1. [Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).](https://arxiv.org/abs/1503.02531)

    """

    def __init__(self):
        pass

    def __call__(self, student_score, teacher_score, T=1):
        return torch.mean(
            F.kl_div(
                F.log_softmax(student_score / T, dim=1),
                F.softmax(teacher_score / T, dim=1),
                reduction="none",
            )
        )
