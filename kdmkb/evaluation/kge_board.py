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

    def update(self, step, model_name, dataset_name, metrics, metadata):
        metadata = f'{self._format_metadata(metadata)}'

        for m, s in metrics.items():

            self.writer.add_scalars(
                main_tag=f'{dataset_name}_{m}',
                tag_scalar_dict={f'{model_name}_{metadata}': s},
                global_step=step
            )

    @classmethod
    def _format_metadata(cls, metadata):
        return ', '.join([f'{xi}: {yi}' for xi, yi in metadata.items()])
