
import torch

from creme import stats

import collections

from ..utils import bar
from ..losses import adversarial


___all___ = ['Pipeline']


class Pipeline:
    """Pipeline to make knowledge graph embeddings easily.

    Parameters:
        dataset (kdmkb.datasets): Dataset.
        model (kdmkb.models): Model.
        sampling (kdmkb.sampling): Negative sampling method.
        max_step (int): Number of iteration to train the model.
        validation (kdmkb.evaluation): Validation process.
        eval_every (int): Interval at which the model will be evaluated on the test / valid
            datasets.
        early_stopping_rounds (int): Stops training when model did not improve scores during
            `early_stopping_rounds` iterations.
        device (str): Device.

    Example:

        >>> from kdmkb import datasets
        >>> from kdmkb import evaluation
        >>> from kdmkb import losses
        >>> from kdmkb import models
        >>> from kdmkb import sampling
        >>> from kdmkb import compose

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> device = 'cpu'

        >>> dataset  = datasets.Wn18rr(batch_size = 2, shuffle = True, seed = 42)

        # Do not reproduce at home the code below this comment.
        # To run tests faster I chosse to select a sub part of test and valid datasets.

        >>> dataset.valid = dataset.valid[:4]
        >>> dataset.test = dataset.test[:4]

        # Ok, now you can reproduce the code below at home.

        >>> sampling = sampling.NegativeSampling(
        ...        size          = 4,
        ...        train_triples = dataset.train,
        ...        entities      = dataset.entities,
        ...        relations     = dataset.relations,
        ...        seed          = 42,
        ... )

        >>> model = models.RotatE(
        ...    n_entity   = dataset.n_entity,
        ...    n_relation = dataset.n_relation,
        ...    gamma      = 3,
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
        ...     max_step   = 10, # Should be set to 80000.
        ...     eval_every = 5,  # Should be set to 2000.
        ...     early_stopping_rounds = 3,
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
        Step: 4.
        <BLANKLINE>
        Validation scores - {'MRR': 0.0001, 'MR': 13861.5, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0, 'MRR_relations': 0.4062, 'MR_relations': 3.5, 'HITS@1_relations': 0.0, 'HITS@3_relations': 0.75, 'HITS@10_relations': 1.0}
        <BLANKLINE>
        Test scores - {'MRR': 0.0001, 'MR': 13842.125, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0, 'MRR_relations': 0.4062, 'MR_relations': 3.5, 'HITS@1_relations': 0.0, 'HITS@3_relations': 0.75, 'HITS@10_relations': 1.0}
        <BLANKLINE>
        <BLANKLINE>
        Step: 9.
        <BLANKLINE>
        Validation scores - {'MRR': 0.0001, 'MR': 13865.75, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0, 'MRR_relations': 0.4062, 'MR_relations': 3.5, 'HITS@1_relations': 0.0, 'HITS@3_relations': 0.75, 'HITS@10_relations': 1.0}
        <BLANKLINE>
        Test scores - {'MRR': 0.0001, 'MR': 13844.375, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0, 'MRR_relations': 0.4062, 'MR_relations': 3.5, 'HITS@1_relations': 0.0, 'HITS@3_relations': 0.75, 'HITS@10_relations': 1.0}
        <BLANKLINE>
        <BLANKLINE>
        Step: 9.
        <BLANKLINE>
        <BLANKLINE>
        Validation scores - {'MRR': 0.0001, 'MR': 13865.75, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0, 'MRR_relations': 0.4062, 'MR_relations': 3.5, 'HITS@1_relations': 0.0, 'HITS@3_relations': 0.75, 'HITS@10_relations': 1.0}
        <BLANKLINE>
        <BLANKLINE>
        Test scores - {'MRR': 0.0001, 'MR': 13844.375, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0, 'MRR_relations': 0.4062, 'MR_relations': 3.5, 'HITS@1_relations': 0.0, 'HITS@3_relations': 0.75, 'HITS@10_relations': 1.0}
        <BLANKLINE>

    """

    def __init__(self, max_step, eval_every=2000, early_stopping_rounds=3, device='cpu'):
        self.eval_every = eval_every
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device

        self.loss = adversarial.Adversarial()
        self.bar = bar.Bar(step=max_step, update_every=20, position=0)
        self.metric_loss = stats.RollingMean(1000)

        self.round_without_improvement_valid = 0
        self.round_without_improvement_test = 0

        self.history_valid = collections.defaultdict(float)
        self.history_test = collections.defaultdict(float)

        self.valid_scores = {}
        self.test_scores = {}

    def learn(self, model, dataset, sampling, optimizer, loss, evaluation=None):

        for step in self.bar():

            positive_sample, weight, mode = next(dataset)

            negative_sample = sampling.generate(
                positive_sample=positive_sample,
                mode=mode
            )

            positive_sample = positive_sample.to(self.device)
            negative_sample = negative_sample.to(self.device)
            weight = weight.to(self.device)

            positive_score = model(positive_sample)
            negative_score = model(negative_sample)

            error = self.loss(positive_score, negative_score, weight)

            error.backward()

            _ = optimizer.step()

            self.metric_loss.update(error.item())

            self.bar.set_description(f'loss: {self.metric_loss.get():4f}')

            if evaluation is not None:

                if (step + 1) % self.eval_every == 0:

                    print(f'\n Step: {step}. \n')

                    if dataset.valid:

                        self.valid_scores = evaluation.eval(
                            model=model, dataset=dataset.valid)

                        self.valid_scores.update(evaluation.eval_relations(
                            model=model, dataset=dataset.test))

                        print(f'Validation scores - {self.valid_scores} \n')

                    if dataset.test:

                        self.test_scores = evaluation.eval(
                            model=model, dataset=dataset.test)

                        self.test_scores.update(evaluation.eval_relations(
                            model=model, dataset=dataset.test))

                        print(f'Test scores - {self.test_scores} \n')

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
                        print(f'\n Early stopping at {step} iteration. \n')
                        print(f'Validation scores - {self.valid_scores} \n')
                        print(f'Test scores - {self.test_scores} \n')
                        return self

        print(f'\n Step: {step}. \n')

        if dataset.valid:
            valid_scores = evaluation.eval(
                model=model, dataset=dataset.valid)
            valid_scores.update(
                evaluation.eval_relations(model=model, dataset=dataset.valid)
            )
            print(f'\n Validation scores - {self.valid_scores} \n')

        if dataset.test:
            test_scores = evaluation.eval(
                model=model, dataset=dataset.test)
            test_scores.update(
                evaluation.eval_relations(model=model, dataset=dataset.test)
            )
            print(f'\n Test scores - {self.test_scores} \n')

        return self
