import collections
import os
import operator

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

        if bigger_is_better:
            self.comparison = operator.le
        else:
            self.comparison = operator.ge

        if self.key_metric:
            self.best_params = collections.defaultdict(lambda: collections.defaultdict(float))


    def update(self, model, step, metrics, **description):
        description = f'{model} {self._description(**description)}'

        for m, s in metrics.items():
            self.writer.add_scalars(main_tag=f'{self.experiment}_{m}',
                tag_scalar_dict={f'{model}': s}, global_step=step)

        if self.key_metric in metrics:
            if self.comparison(self.best_params[f'{self.experiment}_{model}'][self.key_metric],
                    metrics[self.key_metric]):

                for m, s in metrics.items():
                    self.best_params[f'{self.experiment}_{model}'][m] = s
                self.best_params[f'{self.experiment}_{model}']['step'] = step


    def export_best_scores(self, model=None):
        '''
        Export best scores founds for each model by default.
        If model is specified, it export the best scores found for this model.
        '''
        if model is not None:

            self.writer.add_text(
                tag         = f'{model}',
                text_string = self._description(**self.best_params[f'{self.experiment}_{model}']),
                global_step = self.best_params[f'{self.experiment}_{model}']['step']
            )

        else:
            for model, scores in self.best_params.items():
                self.writer.add_text(
                    tag         = model,
                    text_string = self._description(**scores),
                    global_step = scores['step']
                )

    @classmethod
    def _description(cls, **x):
        return ', '.join([f'{xi}: {yi}' for xi, yi in x.items()])
