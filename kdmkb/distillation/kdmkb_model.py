from .distillation import Distillation
from ..evaluation import Evaluation
from ..evaluation import KGEBoard
from ..losses import Adversarial
from ..models import TransE
from ..sampling import NegativeSampling
from .top_k_sampling import TopKSampling
from .top_k_sampling import TopKSamplingTransE
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

        Example:

        >>> from kdmkb import models
        >>> from kdmkb import datasets
        >>> from kdmkb import distillation

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> device = 'cpu'

        >>> max_step = 2

        >>> dataset_1 = datasets.Wn18rr(batch_size = 2, seed = 42)
        >>> dataset_1.valid = dataset_1.valid[:2]
        >>> dataset_1.test = dataset_1.test[:2]

        >>> dataset_2 = datasets.Wn18rr(batch_size = 2, seed = 42)
        >>> dataset_2.valid = dataset_2.valid[:2]
        >>> dataset_2.test = dataset_2.test[:2]

        >>> model_1 = models.TransE(
        ...     hidden_dim = 3,
        ...     n_entity = dataset_1.n_entity,
        ...     n_relation = dataset_1.n_relation,
        ...     gamma = 3
        ... ).to(device)

        >>> model_2 = models.RotatE(
        ...     hidden_dim = 3,
        ...     n_entity = dataset_2.n_entity,
        ...     n_relation = dataset_2.n_relation,
        ...     gamma = 3
        ... ).to(device)

        >>> alpha_kl = 0.98

        >>> lr = {'dataset_1': 0.00005, 'dataset_2': 0.00005}
        >>> alpha_adv = {'dataset_1': 0.5, 'dataset_2': 0.5}
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
        ...     eval_every = 2,
        ...     update_every = 1
        ... )
        <BLANKLINE>
        Model: dataset_1, step 1
            Validation:
                    valid_MRR: 0.0001
                    valid_MR: 22704.5
                    valid_HITS@1: 0.0
                    valid_HITS@3: 0.0
                    valid_HITS@10: 0.0
            Test:
                    test_MRR: 0.0007
                    test_MR: 8102.5
                    test_HITS@1: 0.0
                    test_HITS@3: 0.0
                    test_HITS@10: 0.0
            Relation:
                    test_MRR_relations: 0.5
                    test_MR_relations: 2.0
                    test_HITS@1_relations: 0.0
                    test_HITS@3_relations: 1.0
                    test_HITS@10_relations: 1.0
        <BLANKLINE>
        Model: dataset_2, step 1
            Validation:
                    valid_MRR: 0.0001
                    valid_MR: 24176.5
                    valid_HITS@1: 0.0
                    valid_HITS@3: 0.0
                    valid_HITS@10: 0.0
            Test:
                    test_MRR: 0.0001
                    test_MR: 15013.25
                    test_HITS@1: 0.0
                    test_HITS@3: 0.0
                    test_HITS@10: 0.0
            Relation:
                    test_MRR_relations: 0.1389
                    test_MR_relations: 7.5
                    test_HITS@1_relations: 0.0
                    test_HITS@3_relations: 0.0
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
                    ) * self.alpha_kl

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
                            model_name=models[id_dataset].name,
                            dataset_name=dataset.name,
                            metadata={
                                'hidden_dim': models[id_dataset].hidden_dim,
                                'gamma': models[id_dataset].gamma.item(),

                            },
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
