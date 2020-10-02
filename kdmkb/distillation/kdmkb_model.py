from ..datasets import Dataset

from .distillation import Distillation

from ..evaluation import Evaluation

from ..losses import Adversarial
from ..losses import BCEWithLogitsLoss

from ..models import TransE

from ..sampling import NegativeSampling

from .top_k_sampling import TopKSampling
from .top_k_sampling import TopKSamplingTransE

from ..utils import BarRange

import collections
import os

from creme import stats

import pickle

import torch

import numpy as np
import pandas as pd


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

        Example:

        >>> from kdmkb import models
        >>> from kdmkb import datasets
        >>> from kdmkb import distillation

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> device = 'cpu'

        >>> max_step = 10

        >>> dataset_1 = datasets.CountriesS1(batch_size = 2, seed = 42)

        >>> dataset_2 = datasets.CountriesS1(batch_size = 2, seed = 42)

        >>> model_1 = models.TransE(
        ...     hidden_dim = 3,
        ...     n_entity = dataset_1.n_entity,
        ...     n_relation = dataset_1.n_relation,
        ...     gamma = 3
        ... ).to(device)

        >>> model_2 = models.ConvE(
        ...     hidden_dim = (5, 5),
        ...     n_entity = dataset_2.n_entity,
        ...     n_relation = dataset_2.n_relation,
        ... ).to(device)

        >>> alpha_kl = {'dataset_1': 0.98, 'dataset_2': 0.98}

        >>> lr = {'dataset_1': 0.00005, 'dataset_2': 0.00005}
        >>> alpha_adv = {'dataset_1': 0.98, 'dataset_2': 0.5}
        >>> negative_sampling_size = {'dataset_1': 3, 'dataset_2': 3}

        >>> batch_size_entity = {'dataset_1': 2, 'dataset_2': 2}
        >>> batch_size_relation = {'dataset_1': 2, 'dataset_2': 2}

        >>> n_random_entities = {'dataset_1': 2, 'dataset_2': 2}
        >>> n_random_relations = {'dataset_1': 2, 'dataset_2': 2}

        >>> kdmkb_model = distillation.KdmkbModel(
        ...    models = {'dataset_1': model_1, 'dataset_2': model_2},
        ...    datasets = {'dataset_1': dataset_1, 'dataset_2': dataset_2},
        ...    lr = lr,
        ...    alpha_kl = alpha_kl,
        ...    alpha_adv = alpha_adv,
        ...    negative_sampling_size = negative_sampling_size,
        ...    batch_size_entity = batch_size_entity,
        ...    batch_size_relation = batch_size_relation,
        ...    n_random_entities = n_random_entities,
        ...    n_random_relations = n_random_relations,
        ...    device = device,
        ...    seed = 42,
        ... )

        >>> kdmkb_model = kdmkb_model.learn(
        ...     models = {'dataset_1': model_1, 'dataset_2': model_2},
        ...     datasets = {'dataset_1': dataset_1, 'dataset_2': dataset_2},
        ...     max_step = max_step,
        ...     eval_every = 10,
        ...     update_every = 3,
        ... )
        <BLANKLINE>
        Model: dataset_1, step 9
            Validation:
                valid_MRR: 0.0143
                valid_MR: 122.4792
                valid_HITS@1: 0.0
                valid_HITS@3: 0.0
                valid_HITS@10: 0.0
                valid_MRR_relations: 0.9375
                valid_MR_relations: 1.125
                valid_HITS@1_relations: 0.875
                valid_HITS@3_relations: 1.0
                valid_HITS@10_relations: 1.0
            Test:
                test_MRR: 0.016
                test_MR: 132.8125
                test_HITS@1: 0.0
                test_HITS@3: 0.0
                test_HITS@10: 0.0417
                test_MRR_relations: 0.9167
                test_MR_relations: 1.1667
                test_HITS@1_relations: 0.8333
                test_HITS@3_relations: 1.0
                test_HITS@10_relations: 1.0
        <BLANKLINE>
        Model: dataset_2, step 9
            Validation:
                valid_MRR: 0.0147
                valid_MR: 142.0625
                valid_HITS@1: 0.0
                valid_HITS@3: 0.0
                valid_HITS@10: 0.0417
                valid_MRR_relations: 0.75
                valid_MR_relations: 1.5
                valid_HITS@1_relations: 0.5
                valid_HITS@3_relations: 1.0
                valid_HITS@10_relations: 1.0
            Test:
                test_MRR: 0.0123
                test_MR: 154.7083
                test_HITS@1: 0.0
                test_HITS@3: 0.0
                test_HITS@10: 0.0417
                test_MRR_relations: 0.7083
                test_MR_relations: 1.5833
                test_HITS@1_relations: 0.4167
                test_HITS@3_relations: 1.0
                test_HITS@10_relations: 1.0

    """

    def __init__(
        self, models, datasets, lr, alpha_kl, alpha_adv, negative_sampling_size, batch_size_entity,
        batch_size_relation, n_random_entities, n_random_relations, device='cuda', seed=None
    ):

        self.alpha_kl = alpha_kl
        self.batch_size_entity = batch_size_entity
        self.batch_size_relation = batch_size_relation
        self.n_random_entities = n_random_entities
        self.n_random_relations = n_random_relations
        self.device = device
        self.seed = seed
        self._rng = np.random.RandomState(  # pylint: disable=no-member
            self.seed)

        self.loss_function = collections.OrderedDict()

        for id_dataset, dataset in datasets.items():

            if dataset.classification:
                self.loss_function[id_dataset] = BCEWithLogitsLoss()

            else:
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

                    if isinstance(models[id_dataset_teacher], TransE):
                        # Sampling for TransE is based on faiss and is faster.
                        sampling_method = TopKSamplingTransE
                    else:
                        sampling_method = TopKSampling

                    self.distillation[
                        f'{id_dataset_teacher}_{id_dataset_student}'
                    ] = self._init_distillation(
                        sampling_method=sampling_method,
                        teacher=models[id_dataset_teacher],
                        dataset_teacher=dataset_teacher,
                        dataset_student=dataset_student,
                        batch_size_entity=self.batch_size_entity[id_dataset_teacher],
                        batch_size_relation=self.batch_size_relation[id_dataset_teacher],
                        n_random_entities=self.n_random_entities[id_dataset_teacher],
                        n_random_relations=self.n_random_relations[id_dataset_teacher],
                        seed=self.seed,
                        device=self.device,
                    )

        self.negative_sampling = collections.OrderedDict()

        self.validation = collections.OrderedDict()

        for id_dataset, dataset in datasets.items():

            if not dataset.classification:

                self.negative_sampling[id_dataset] = NegativeSampling(
                    size=negative_sampling_size[id_dataset],
                    entities=dataset.entities,
                    relations=dataset.relations,
                    train_triples=dataset.train_triples,
                    seed=seed
                )

            self.validation[id_dataset] = Evaluation(
                entities=dataset.entities,
                relations=dataset.relations,
                batch_size=2,
                true_triples=dataset.true_triples,
                device=device,
            )

        self.metrics = {
            id_dataset: stats.RollingMean(1000) for id_dataset, _ in datasets.items()
        }

    @classmethod
    def _init_distillation(
        cls, sampling_method, teacher, dataset_teacher, dataset_student, batch_size_entity,
        batch_size_relation, n_random_entities, n_random_relations, seed, device
    ):
        return Distillation(
            teacher_entities=dataset_teacher.entities,
            teacher_relations=dataset_teacher.relations,
            student_entities=dataset_student.entities,
            student_relations=dataset_student.relations,
            sampling=sampling_method(**{
                'teacher': teacher,
                'teacher_relations': dataset_teacher.relations,
                'teacher_entities': dataset_teacher.entities,
                'student_entities': dataset_student.entities,
                'student_relations': dataset_student.relations,
                'batch_size_entity': batch_size_entity,
                'batch_size_relation': batch_size_relation,
                'n_random_entities': n_random_entities,
                'n_random_relations': n_random_relations,
                'seed': seed,
                'device': device,
            }),
            device=device,
        )

    def forward(self, datasets, models):

        loss_models = collections.OrderedDict()
        samples = collections.OrderedDict()

        for id_dataset, dataset in datasets.items():

            data = next(dataset)

            sample = data['sample']

            mode = data['mode']

            scores = models[id_dataset](sample)

            if mode == 'classification':

                sample = sample.to(self.device)

                y = data['y'].to(self.device)

                loss_models[id_dataset] = self.loss_function[id_dataset](
                    scores,
                    y
                ) * (1 - self.alpha_kl[id_dataset])

                sample = self._format_batch_distillation(
                    rng=self._rng,
                    sample=sample,
                    y=y
                )

            else:

                negative_sample = self.negative_sampling[id_dataset].generate(
                    sample=sample,
                    mode=mode,
                )

                weight = data['weight'].to(self.device)

                sample = sample.to(self.device)

                negative_sample = negative_sample.to(self.device)

                negative_score = models[id_dataset](
                    sample,
                    negative_sample,
                    mode=mode
                )

                loss_models[id_dataset] = self.loss_function[id_dataset](
                    positive_score=scores,
                    negative_score=negative_score,
                    weight=weight,
                ) * (1 - self.alpha_kl[id_dataset])

            # Store positive sample to distill it.
            samples[id_dataset] = sample

        for id_dataset_teacher, _ in datasets.items():

            for id_dataset_student, _, in datasets.items():

                if id_dataset_teacher != id_dataset_student:

                    loss_models[id_dataset_student] += self.distillation[
                        f'{id_dataset_teacher}_{id_dataset_student}'
                    ].distill(
                        teacher=models[id_dataset_teacher],
                        student=models[id_dataset_student],
                        sample=samples[id_dataset_teacher]
                    ) * self.alpha_kl[id_dataset]

        for id_dataset, _ in datasets.items():

            loss_models[id_dataset].backward()

            self.optimizers[id_dataset].step()

            self.optimizers[id_dataset].zero_grad()

            self.metrics[id_dataset].update(loss_models[id_dataset].item())

        # Update distillation when using faiss trees.
        for id_dataset_teacher, dataset_teacher in datasets.items():

            for id_dataset_student, dataset_student in datasets.items():

                if id_dataset_teacher != id_dataset_student:

                    if isinstance(models[id_dataset_teacher], TransE):

                        self.distillation[
                            f'{id_dataset_teacher}_{id_dataset_student}'
                        ] = self._init_distillation(
                            sampling_method=TopKSamplingTransE,
                            teacher=models[id_dataset_teacher],
                            dataset_teacher=dataset_teacher,
                            dataset_student=dataset_student,
                            batch_size_entity=self.batch_size_entity[id_dataset_teacher],
                            batch_size_relation=self.batch_size_relation[id_dataset_teacher],
                            n_random_entities=self.n_random_entities[id_dataset_teacher],
                            n_random_relations=self.n_random_relations[id_dataset_teacher],
                            seed=self.seed,
                            device=self.device,
                        )

        return self.metrics

    def learn(self, models, datasets, max_step, eval_every=2000,
              update_every=10, log_dir=None, save_path=None):
        """
            Parameters:
                models: (dict[id_dataset, kdmkb.models]): Mapping between ids datasets and models.
                datasets (dict[id_dataset, kdmkb.datasets]): Mapping between ids datasets and datasets.
                max_step (int): Number of steps to train the model.
                eval_every (int): Eval each models of kdmkb every `eval_every` steps.
                update_every (int): Update tqdm bar description every `update_every` steps.
                log_dir (str): Path to export evaluation scores as pandas DataFrame.
                save_path (str): Path to pickle the model.

        """

        # Load existing scores
        if log_dir is not None:

            scores = []

            if os.path.isfile(log_dir):

                scores.append(pd.read_csv(f'{log_dir}'))

        bar = BarRange(step=max_step, update_every=update_every)

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

                    scores_valid.update(self.validation[id_dataset].eval_relations(
                        model=models[id_dataset],
                        dataset=dataset.valid
                    ))

                    scores_valid = collections.OrderedDict({
                        f'valid_{metric}': score for metric, score in scores_valid.items()
                    })

                    scores_test = self.validation[id_dataset].eval(
                        model=models[id_dataset],
                        dataset=dataset.test
                    )

                    scores_test.update(self.validation[id_dataset].eval_relations(
                        model=models[id_dataset],
                        dataset=dataset.test
                    ))

                    scores_test = collections.OrderedDict({
                        f'test_{metric}': score for metric, score in scores_test.items()
                    })

                    models[id_dataset] = models[id_dataset].train()

                    print(f'\n Model: {id_dataset}, step {step}')

                    self.print_metrics(
                        description='Validation:',
                        metrics=scores_valid
                    )

                    self.print_metrics(
                        description='Test:',
                        metrics=scores_test
                    )

                    # Export results to pandas dataframe
                    if log_dir is not None:

                        scores.append(pd.concat(
                            [
                                pd.DataFrame.from_dict({
                                    'dataset': dataset.name,
                                    'model_name': models[id_dataset].name,
                                    'step': step,
                                }, orient='index').T,
                                pd.DataFrame.from_dict({
                                    'batch_size': dataset.batch_size,
                                    'negative_sample_size': self.negative_sampling[id_dataset].size
                                    if not dataset.classification else 0,
                                    'hidden_dim': models[id_dataset].hidden_dim,
                                    'gamma': models[id_dataset].gamma.item()
                                    if not dataset.classification else 0,
                                }, orient='index').T,
                                pd.DataFrame.from_dict(
                                    scores_valid, orient='index').T,
                                pd.DataFrame.from_dict(
                                    scores_test, orient='index').T,
                            ],
                            axis='columns'
                        ))

                        pd.concat(scores, axis='rows').reset_index(
                            drop=True).to_csv(log_dir, index=False)

                    # Save models
                    if save_path is not None:

                        model_name = f'kdmkb_{dataset.name}_{id_dataset}_{models[id_dataset].name}_{step}.pickle'

                        models[id_dataset].cpu().save(
                            path=os.path.join(save_path, model_name))

                        models[id_dataset] = models[id_dataset].to(self.device)

        return self

    @classmethod
    def print_metrics(cls, description, metrics):
        print(f'\t {description}')
        for metric, value in metrics.items():
            print(f'\t\t {metric}: {value}')

    @staticmethod
    def _format_batch_distillation(rng, sample, y):
        """When using classification dataset, select a random tail among existing tails for a given
        head and relation in the dataset to complete sample for distillation.
        """
        tails = []
        for x in y.detach():
            index = (x == 1.).nonzero().flatten().numpy()
            tails.append(rng.choice(index))
        return torch.cat([
            sample,
            torch.tensor(tails).unsqueeze(1),
        ], dim=1)
