import torch.nn as nn


__all__ = ['BCEWithLogitsLoss']


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def __init__(self):
        super().__init__()
