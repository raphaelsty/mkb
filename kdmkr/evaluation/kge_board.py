import collections
import os

import torch
from torch.utils.tensorboard import SummaryWriter


__all__ = ['KGEBoard']


class KGEBoard:
    '''
    KGEBoard: Tensorboard interface.

    '''
    def __init__(self, log_dir, experiment, key_metric='HITS@3', bigger_is_better=True):
        self.experiment       = experiment
        self.path             = os.path.join(log_dir, experiment)
        self.writer           = SummaryWriter(log_dir=self.path)
        self.key_metric       = key_metric
        self.bigger_is_better = bigger_is_better

        if self.key_metric:
            self.best_params = collections.defaultdict(lambda: collections.defaultdict(float))

    def update(self, model, step, metrics, prefix=''):
        description = f'{model} {self._description(x=metrics)}'

        for m, s in metrics.items():
            self.writer.add_scalars(
                main_tag=f'{prefix}_{m}', tag_scalar_dict={f'{model}': s}, global_step=step)

        # Write on tensorboard best models
        if self.key_metric and self.best_params[model][self.key_metric] < metrics[self.key_metric]:
            self.best_params.pop(model)
            for m, s in metrics.items():
                self.best_params[f'{self.experiment}_{model}'][m] = s
            self.best_params[f'{self.experiment}_{model}']['step'] = step

    @classmethod
    def _description(cls, x):
        return ', '.join([f'{xi}: {yi}' for xi, yi in x.items()])


    def export_best_params(self):
        for model, scores in self.best_params.items():
            self.writer.add_text(model, str(dict(scores)), global_step = scores['step'])
