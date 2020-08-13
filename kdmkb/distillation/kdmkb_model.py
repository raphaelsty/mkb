from .distillation import Distillation
from .top_k_sampling import TopKSampling
from ..sampling import NegativeSampling
from ..losses import Adversarial
from ..evaluation import Evaluation
from ..evaluation import KGEBoard

from ..utils import Bar

import collections
import os

from creme import stats

import pickle

import torch


__all__ = ['KdmkbModel']


class KdmkbModel:
    """Training model of kdmkb.

    Parameters:

        models (dict[str, kdmkb.models]): Mapping of datasets ids with corresponding models.
        datasets (dict[str, kdmkb.datasets]): Mapping of datasets id with corresponding datasets.
        lr (dict[str, float]): Mapping of datasets ids with corresponding learning rate.
        alpha_kl (float): Weight of the distillation.
        alpha_adv (float): Alpha coefficient of adversarial loss.
        negative_sampling_size (dict[str, int]): Mapping of datasets ids with corresponding negative
            sample size.
        batch_size_entity (dict[str, int]): Number of entities to use in distillation.
        batch_size_relations (dict[str, int]): Number of relations to use in distillation.
        n_random_entities (dict[str, int]): Number of random entities to use in distillation.
        n_random_relations (dict[str, int]): Number of random relations to use in distillation.
        device (str): Device to use, cuda or cpu.
        seed (int): Random state.

    """

    def __init__(
        self, models, datasets, lr, alpha_kl, alpha_adv, negative_sampling_size, batch_size_entity,
        batch_size_relation, n_random_entities, n_random_relations, device='cuda', seed=None
    ):

        self.alpha_kl = alpha_kl

        self.loss_function = collections.OrderedDict()

        for id_dataset, _ in datasets.items():

            self.loss_function[id_dataset] = Adversarial(
                alpha=alpha_adv[id_dataset]
            )

        self.optimizers = collections.OrderedDict()
        for id_dataset, learning_rate in lr.items():
            self.optimizers[id_dataset] = torch.optim.Adam(
                filter(lambda p:
                       p.requires_grad,
                       models[id_dataset].parameters()),
                lr=learning_rate
            )

        self.distillation = collections.OrderedDict()

        for id_dataset_teacher, dataset_teacher in datasets.items():

            for id_dataset_student, dataset_student in datasets.items():

                if id_dataset_teacher != id_dataset_student:

                    self.distillation[
                        f'{id_dataset_teacher}_{id_dataset_student}'
                    ] = Distillation(
                        teacher_entities=dataset_teacher.entities,
                        teacher_relations=dataset_teacher.relations,
                        student_entities=dataset_student.entities,
                        student_relations=dataset_student.relations,
                        sampling=TopKSampling(
                            teacher_relations=dataset_teacher.relations,
                            teacher_entities=dataset_teacher.entities,
                            student_entities=dataset_student.entities,
                            student_relations=dataset_student.relations,
                            batch_size_entity=batch_size_entity[id_dataset_teacher],
                            batch_size_relation=batch_size_relation[id_dataset_teacher],
                            n_random_entities=n_random_entities[id_dataset_teacher],
                            n_random_relations=n_random_relations[id_dataset_teacher],
                            seed=seed,
                            device=device,
                        ),
                        device=device,
                    )

        self.device = device

        self.negative_sampling = collections.OrderedDict()
        self.validation = collections.OrderedDict()

        for id_dataset, dataset in datasets.items():

            self.negative_sampling[id_dataset] = NegativeSampling(
                size=negative_sampling_size[id_dataset],
                entities=dataset.entities,
                relations=dataset.relations,
                train_triples=dataset.train,
                seed=seed
            )

            self.validation[id_dataset] = Evaluation(
                entities=dataset.entities,
                relations=dataset.relations,
                batch_size=2,
                true_triples=dataset.train + dataset.valid + dataset.test,
                device=device,
            )

        self.metrics = {
            id_dataset: stats.RollingMean(1000) for id_dataset, _ in datasets.items()
        }

    def forward(self, datasets, models):

        loss_models = collections.OrderedDict()
        positive_samples = collections.OrderedDict()

        for id_dataset, dataset in datasets.items():

            positive_sample, weight, mode = next(dataset)

            negative_sample = self.negative_sampling[id_dataset].generate(
                positive_sample=positive_sample,
                mode=mode,
            )

            positive_sample = positive_sample.to(self.device)
            negative_sample = negative_sample.to(self.device)
            weight = weight.to(self.device)

            # Store positive sample to distill it.
            positive_samples[id_dataset] = positive_sample

            positive_score = models[id_dataset](positive_sample)

            negative_score = models[id_dataset](
                positive_sample,
                negative_sample,
                mode=mode
            )

            loss_models[id_dataset] = self.loss_function[id_dataset](
                positive_score=positive_score,
                negative_score=negative_score,
                weight=weight,
            ) * (1 - self.alpha_kl)

        for id_dataset_teacher, _ in datasets.items():

            for id_dataset_student, _, in datasets.items():

                if id_dataset_teacher != id_dataset_student:

                    loss_models[id_dataset_student] += self.distillation[
                        f'{id_dataset_teacher}_{id_dataset_student}'
                    ].distill(
                        teacher=models[id_dataset_teacher],
                        student=models[id_dataset_student],
                        positive_sample=positive_samples[id_dataset_teacher]
                    )

        for id_dataset, _ in datasets.items():

            loss_models[id_dataset].backward()

            self.optimizers[id_dataset].step()

            self.optimizers[id_dataset].zero_grad()

            self.metrics[id_dataset].update(loss_models[id_dataset].item())

        return self.metrics

    def learn(self, models, datasets, max_step, eval_every=2000,
              update_every=10, log_dir=None, experiment=None, save_path=None):

        if log_dir is not None and experiment is not None:

            board = KGEBoard(
                log_dir=log_dir,
                experiment=experiment,
            )

        bar = Bar(step=max_step, update_every=update_every)

        for step in bar:

            metrics = self.forward(datasets, models)

            bar.set_description(
                text=', '.join(
                    [f'{model}: {loss}' for model, loss in metrics.items()]
                )
            )

            if (step + 1) % eval_every == 0:

                for id_dataset, dataset in datasets.items():

                    models[id_dataset] = models[id_dataset].eval()

                    scores_valid = self.validation[id_dataset].eval(
                        model=models[id_dataset],
                        dataset=dataset.valid
                    )

                    scores_valid = collections.OrderedDict({
                        f'valid_{metric}': score for metric, score in scores_valid.items()
                    })

                    scores_test = self.validation[id_dataset].eval(
                        model=models[id_dataset],
                        dataset=dataset.test
                    )

                    scores_test = collections.OrderedDict({
                        f'test_{metric}': score for metric, score in scores_test.items()
                    })

                    scores_relation = self.validation[id_dataset].eval_relations(
                        model=models[id_dataset],
                        dataset=dataset.test
                    )

                    models[id_dataset] = models[id_dataset].train()

                    scores_relations_test = collections.OrderedDict({
                        f'test_{metric}': score for metric, score in scores_relation.items()
                    })

                    print(f'\n Model: {id_dataset}, step {step}')

                    self.print_metrics(
                        description='Validation:',
                        metrics=scores_valid
                    )

                    self.print_metrics(
                        description='Test:',
                        metrics=scores_test
                    )

                    self.print_metrics(
                        description='Relation:',
                        metrics=scores_relations_test
                    )

                    # Export results to tensorboard
                    if log_dir is not None and experiment is not None:

                        board.update(
                            step=step,
                            metrics=dict(
                                **scores_valid, **scores_test,
                                **scores_relations_test
                            ),
                            model_id=id_dataset,
                        )

                        # Save models
                    if save_path is not None:

                        scores_to_str = ', '.join(
                            [f'{metric}: {x}' for metric,
                                x in scores_valid.items()]
                        )

                        name_model = f'{id_dataset}_{scores_to_str}.pickle'

                        with open(os.path.join(save_path, name_model), 'wb') as handle:
                            pickle.dump(
                                models[id_dataset],
                                handle,
                                protocol=pickle.HIGHEST_PROTOCOL
                            )
        return self

    @classmethod
    def print_metrics(cls, description, metrics):
        print(f'\t {description}')
        for metric, value in metrics.items():
            print(f'\t\t {metric}: {value}')
