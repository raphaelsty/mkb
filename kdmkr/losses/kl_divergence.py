import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['KlDivergence']


class KlDivergence(nn.Module):
    """Distillation loss."""

    def __init__(self):
        pass

    def __call__(self, student_score, teacher_score):
        return torch.mean(F.kl_div(
            F.log_softmax(student_score, dim=1),
            F.softmax(teacher_score, dim=1),
            reduction='none')
        )
