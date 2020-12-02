
import torch

from creme import stats

import collections

from ..losses import Adversarial
from ..losses import BCEWithLogitsLoss
from ..utils import Bar


___all___ = ['Pipeline']


class Pipeline:
    """Pipeline dedicated to automate training model.

    Parameters:
        dataset (mkb.datasets): Dataset.
        model (mkb.models): Model.
        sampling (mkb.sampling): Negative sampling method.
        epochs (int): Number of epochs to train the model.
        validation (mkb.evaluation): Validation process.
        eval_every (int): When eval_every is set to 1, the model will be evaluated at every epochs.
        early_stopping_rounds (int): Stops training when model did not improve scores during
            `early_stopping_rounds` epochs.
        device (str): Device.

    Example:

        >>> from mkb import datasets
        >>> from mkb import evaluation
        >>> from mkb import losses
        >>> from mkb import models
        >>> from mkb import sampling
        >>> from mkb import compose

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> device = 'cpu'

        >>> train = [
        ...     (0, 0, 1),
        ...     (0, 1, 1),
        ...     (2, 0, 3),
        ...     (2, 1, 3),
        ... ]

        >>> valid = [
        ...     (0, 0, 1),
        ...     (2, 1, 3),
        ... ]

        >>> test = [
        ...     (0, 0, 1),
        ...     (2, 1, 3),
        ... ]

        >>> entities = {
        ... 'e0': 0,
        ... 'e1': 1,
        ... 'e2': 2,
        ... 'e3': 3,
        ... }

        >>> relations = {
        ... 'r0': 0,
        ... 'r1': 1,
        ... }

        >>> dataset = datasets.Dataset(
        ...    train = train,
        ...    valid = valid,
        ...    test = test,
        ...    entities = entities,
        ...    relations = relations,
        ...    batch_size = 2,
        ...    seed = 42,
        ...    shuffle = False,
        ... )

        >>> dataset = datasets.CountriesS1(batch_size = 20, seed = 42)

        >>> sampling = sampling.NegativeSampling(
        ...        size          = 4,
        ...        train_triples = dataset.train,
        ...        entities      = dataset.entities,
        ...        relations     = dataset.relations,
        ...        seed          = 42,
        ... )

        >>> model = models.RotatE(
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    gamma = 3,
        ...    hidden_dim = 5,
        ... )

        >>> model = model.to(device)

        >>> optimizer = torch.optim.Adam(
        ...     filter(lambda p: p.requires_grad, model.parameters()),
        ...     lr = 0.00005,
        ... )

        >>> evaluation = evaluation.Evaluation(
        ...    true_triples = (
        ...       dataset.train +
        ...       dataset.valid +
        ...       dataset.test
        ...    ),
        ...    entities   = dataset.entities,
        ...    relations  = dataset.relations,
        ...    batch_size = 8,
        ...    device     = device,
        ... )

        >>> pipeline = compose.Pipeline(
        ...     epochs = 3,
        ...     eval_every = 3,
        ...     early_stopping_rounds = 2,
        ...     device = 'cpu',
        ... )

        >>> pipeline = pipeline.learn(
        ...     model      = model,
        ...     dataset    = dataset,
        ...     evaluation = evaluation,
        ...     sampling   = sampling,
        ...     optimizer  = optimizer,
        ...     loss       = losses.Adversarial(alpha=0.5),
        ... )
        <BLANKLINE>
        Epoch: 2.
            Validation:
                    MRR: 0.0561
                    MR: 117.6875
                    HITS@1: 0.0417
                    HITS@3: 0.0417
                    HITS@10: 0.0417
                    MRR_relations: 0.7917
                    MR_relations: 1.4167
                    HITS@1_relations: 0.5833
                    HITS@3_relations: 1.0
                    HITS@10_relations: 1.0
            Test:
                    MRR: 0.0089
                    MR: 138.25
                    HITS@1: 0.0
                    HITS@3: 0.0
                    HITS@10: 0.0
                    MRR_relations: 0.6875
                    MR_relations: 1.625
                    HITS@1_relations: 0.375
                    HITS@3_relations: 1.0
                    HITS@10_relations: 1.0
        <BLANKLINE>
            Epoch: 2.
        <BLANKLINE>
                Validation:
                        MRR: 0.0561
                        MR: 117.6875
                        HITS@1: 0.0417
                        HITS@3: 0.0417
                        HITS@10: 0.0417
                        MRR_relations: 0.7917
                        MR_relations: 1.4167
                        HITS@1_relations: 0.5833
                        HITS@3_relations: 1.0
                        HITS@10_relations: 1.0
                Test:
                        MRR: 0.0089
                        MR: 138.25
                        HITS@1: 0.0
                        HITS@3: 0.0
                        HITS@10: 0.0
                        MRR_relations: 0.6875
                        MR_relations: 1.625
                        HITS@1_relations: 0.375
                        HITS@3_relations: 1.0
                        HITS@10_relations: 1.0


    """

    def __init__(self, epochs, eval_every=2000, early_stopping_rounds=3, device='cpu'):
        self.epochs = epochs
        self.eval_every = eval_every
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device

        self.metric_loss = stats.RollingMean(1000)

        self.round_without_improvement_valid = 0
        self.round_without_improvement_test = 0

        self.history_valid = collections.defaultdict(float)
        self.history_test = collections.defaultdict(float)

        self.valid_scores = {}
        self.test_scores = {}

    def learn(self, model, dataset, sampling, optimizer, loss, evaluation=None):

        for epoch in range(self.epochs):

            bar = Bar(dataset=dataset, update_every=10)

            for data in bar:

                sample = data['sample'].to(self.device)
                mode = data['mode']

                score = model(sample)

                if mode == 'classification':

                    y = data['y'].to(self.device)

                    error = loss(score, y)

                else:

                    weight = data['weight'].to(self.device)

                    negative_sample = sampling.generate(
                        sample=sample,
                        mode=mode,
                    )

                    negative_sample = negative_sample.to(self.device)

                    negative_score = model(
                        sample=sample,
                        negative_sample=negative_sample,
                        mode=mode
                    )

                    error = loss(score, negative_score, weight)

                error.backward()

                _ = optimizer.step()

                optimizer.zero_grad()

                self.metric_loss.update(error.item())

                bar.set_description(
                    f'Epoch: {epoch}, loss: {self.metric_loss.get():4f}')

            if evaluation is not None:

                if (epoch + 1) % self.eval_every == 0:

                    print(f'\n Epoch: {epoch}.')

                    if dataset.valid:

                        self.valid_scores = evaluation.eval(
                            model=model, dataset=dataset.valid)

                        self.valid_scores.update(evaluation.eval_relations(
                            model=model, dataset=dataset.valid))

                        self.print_metrics(
                            description='Validation:',
                            metrics=self.valid_scores
                        )

                    if dataset.test:

                        self.test_scores = evaluation.eval(
                            model=model, dataset=dataset.test)

                        self.test_scores.update(evaluation.eval_relations(
                            model=model, dataset=dataset.test))

                        self.print_metrics(
                            description='Test:',
                            metrics=self.test_scores
                        )

                        if (self.history_test['HITS@3'] > self.test_scores['HITS@3'] and
                                self.history_test['HITS@1'] > self.test_scores['HITS@1']):
                            self.round_without_improvement_test += 1
                        else:
                            self.round_without_improvement_test = 0
                            self.history_test = self.test_scores
                    else:
                        if (self.history_valid['HITS@3'] > self.valid_scores['HITS@3'] and
                                self.history_valid['HITS@1'] > self.valid_scores['HITS@1']):
                            self.round_without_improvement_valid += 1
                        else:
                            self.round_without_improvement_valid = 0
                            self.history_valid = self.valid_scores

                    if (self.round_without_improvement_valid == self.early_stopping_rounds or
                            self.round_without_improvement_test == self.early_stopping_rounds):

                        print(
                            f'\n Early stopping at epoch {epoch}.')

                        self.print_metrics(
                            description='Validation:',
                            metrics=self.valid_scores
                        )

                        self.print_metrics(
                            description='Test:',
                            metrics=self.test_scores
                        )

                        return self

        print(f'\n Epoch: {epoch}. \n')

        if dataset.valid:

            self.valid_scores = evaluation.eval(
                model=model, dataset=dataset.valid)

            self.valid_scores.update(
                evaluation.eval_relations(model=model, dataset=dataset.valid)
            )

            self.print_metrics(
                description='Validation:',
                metrics=self.valid_scores
            )

        if dataset.test:

            self.test_scores = evaluation.eval(
                model=model, dataset=dataset.test)

            self.test_scores.update(
                evaluation.eval_relations(model=model, dataset=dataset.test)
            )

            self.print_metrics(
                description='Test:',
                metrics=self.test_scores
            )

        return self

    @classmethod
    def print_metrics(cls, description, metrics):
        print(f'\t {description}')
        for metric, value in metrics.items():
            print(f'\t\t {metric}: {value}')
