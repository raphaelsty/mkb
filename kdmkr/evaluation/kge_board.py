import os

import torch

from torch.utils.tensorboard import SummaryWriter


__all__ = ['KGEBoard']


class KGEBoard:
    '''
    KGEBoard: Tensorboard interface.

    '''
    def __init__(self, log_dir, experiment):
        self.path = os.path.join(log_dir, experiment)
        self.writer = SummaryWriter(log_dir=self.path)

    def update(self, model, step, metrics):
        description = f'{model} {self._description(x=metrics)}'

        for m, s in metrics.items():
            self.writer.add_scalars(
                main_tag=f'{m}', tag_scalar_dict={f'{model}': s}, global_step=step)

    @classmethod
    def _description(cls, x):
        return ', '.join([f'{xi}: {yi}' for xi, yi in x.items()])
