import collections
import os
import operator

import torch


__all__ = ['KGEBoard']


class KGEBoard:
    """
    KGEBoard: Tensorboard interface.

    """

    def __init__(self, log_dir, experiment):
        from torch.utils.tensorboard import SummaryWriter

        self.experiment = experiment
        self.path = os.path.join(log_dir, experiment)
        self.writer = SummaryWriter(log_dir=self.path)

    def update(self, step, metrics, model_id):
        model_id = f'{self._model_id(model_id)}'

        for m, s in metrics.items():

            self.writer.add_scalars(
                main_tag=f'{self.experiment}_{m}',
                tag_scalar_dict={f'{model_id}': s},
                global_step=step
            )

    @classmethod
    def _model_id(cls, model_id):
        return ', '.join([f'{xi}: {yi}' for xi, yi in model_id.items()])
