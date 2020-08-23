from ..datasets import base

import collections

from creme import stats

from torch.utils import data

import torch

import pandas as pd


__all__ = ['Evaluation']


class Evaluation:
    """Evaluate model on selected dataset.

    Evaluate metrics Hits@1, Hits@3, Hits@10, MRR, MR on entities, relations and tails.
    Returns distincts metrics for link prediction ie, entities or relations and relation
    predictions.

    Parameters:
        entities (dict): Entities of the dataset.
        relations (dict): Relations of the dataset.
        batch_size (int): Size of the batch.
        true_triples (list): Available triplets to filter metrics. If not specified, `Evaluation`
            will mesure raw metrics. Usually we filter triplets based on train, validation and test
            datasets.
        device (str): cpu or cuda.
        num_workers (str): Number of workers for pytorch dataset.

    Example:

        >>> from kdmkb import datasets
        >>> from kdmkb import evaluation
        >>> from kdmkb import models
        >>> from kdmkb import losses
        >>> from kdmkb import sampling

        >>> import torch

        >>> _ = torch.manual_seed(42)

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

        >>> dataset = datasets.Fetch(
        ...    train = train,
        ...    test = test,
        ...    entities = entities,
        ...    relations = relations,
        ...    batch_size = 2,
        ...    seed = 42,
        ...    shuffle = False,
        ... )

        >>> negative_sampling = sampling.NegativeSampling(
        ...    size = 2,
        ...    train_triples = dataset.train,
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    seed = 42,
        ... )

        >>> model = models.RotatE(hidden_dim=3, n_entity=dataset.n_entity,
        ...    n_relation=dataset.n_relation, gamma=1)

        >>> optimizer = torch.optim.Adam(
        ...    filter(lambda p: p.requires_grad, model.parameters()),
        ...    lr = 0.5,
        ... )

        >>> loss = losses.Adversarial(alpha=0.5)

        >>> for _ in range(10):
        ...     positive_sample, weight, mode=next(dataset)
        ...     positive_score = model(positive_sample)
        ...     negative_sample = negative_sampling.generate(positive_sample=positive_sample,
        ...         mode=mode)
        ...     negative_score = model(positive_sample, negative_sample, mode)
        ...     loss(positive_score, negative_score, weight).backward()
        ...     _ = optimizer.step()

        >>> model = model.eval()

        >>> validation = evaluation.Evaluation(true_triples=train + valid + test,
        ...     entities=entities, relations=relations, batch_size=2)

        >>> validation.eval(model=model, dataset=test)
        {'MRR': 0.6875, 'MR': 2.0, 'HITS@1': 0.5, 'HITS@3': 0.75, 'HITS@10': 1.0}

        >>> validation.eval_relations(model=model, dataset=test)
        {'MRR_relations': 1.0, 'MR_relations': 1.0, 'HITS@1_relations': 1.0,
            'HITS@3_relations': 1.0, 'HITS@10_relations': 1.0}

        >>> validation.detail_eval(model=model, dataset=test, threshold=1.5)
                    head                             tail                             metadata
                    MRR   MR HITS@1 HITS@3 HITS@10   MRR   MR HITS@1 HITS@3 HITS@10 frequency
        relation
        1_1       0.000  0.0    0.0    0.0     0.0  0.00  0.0    0.0    0.0     0.0       0.0
        1_M       0.000  0.0    0.0    0.0     0.0  0.00  0.0    0.0    0.0     0.0       0.0
        M_1       0.000  0.0    0.0    0.0     0.0  0.00  0.0    0.0    0.0     0.0       0.0
        M_M       0.625  2.5    0.5    0.5     1.0  0.75  1.5    0.5    1.0     1.0       1.0

        >>> validation.types_relations(model = model, dataset=test, threshold=1.5)
        {0: 'M_M', 1: 'M_M'}

    References:
        1. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, entities, relations, batch_size, true_triples=[], device='cpu', num_workers=1):
        self.entities = entities
        self.relations = relations
        self.true_triples = true_triples
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers

    def _get_test_loader(self, triples, mode):
        test_dataset = base.TestDataset(
            triples=triples, true_triples=self.true_triples, entities=self.entities,
            relations=self.relations, mode=mode)

        return data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=base.TestDataset.collate_fn)

    def get_entity_stream(self, dataset):
        """Get stream dedicated to link prediction."""
        head_loader = self._get_test_loader(triples=dataset, mode='head-batch')
        tail_loader = self._get_test_loader(triples=dataset, mode='tail-batch')
        return [head_loader, tail_loader]

    def get_relation_stream(self, dataset):
        """Get stream dedicated to relation prediction."""
        test_dataset = base.TestDatasetRelation(
            triples=dataset, true_triples=self.true_triples, entities=self.entities,
            relations=self.relations)

        return data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=base.TestDatasetRelation.collate_fn)

    def eval(self, model, dataset):
        """Evaluate selected model with the metrics: MRR, MR, HITS@1, HITS@3, HITS@10"""
        metrics = collections.OrderedDict({
            metric: stats.Mean()
            for metric in ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']
        })

        with torch.no_grad():

            for test_set in self.get_entity_stream(dataset):

                metrics = self.compute_score(
                    model=model,
                    test_set=test_set,
                    metrics=metrics,
                    device=self.device
                )

        return {name: round(metric.get(), 4) for name, metric in metrics.items()}

    def eval_relations(self, model, dataset):
        metrics = collections.OrderedDict({
            f'{metric}': stats.Mean()
            for metric in ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']
        })

        with torch.no_grad():

            metrics = self.compute_score(
                model=model,
                test_set=self.get_relation_stream(dataset),
                metrics=metrics,
                device=self.device
            )

        return {f'{name}_relations': round(metric.get(), 4) for name, metric in metrics.items()}

    @classmethod
    def compute_score(cls, model, test_set, metrics, device):

        for step, (positive_sample, negative_sample, filter_bias, mode) in enumerate(test_set):

            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            filter_bias = filter_bias.to(device)

            if mode == 'head-batch' or mode == 'tail-batch':

                score = model(
                    sample=positive_sample,
                    negative_sample=negative_sample,
                    mode=mode
                )

            elif mode == 'relation-batch':

                score = model(negative_sample)

            score += filter_bias

            argsort = torch.argsort(score, dim=1, descending=True)

            if mode == 'head-batch':
                positive_arg = positive_sample[:, 0]

            if mode == 'relation-batch':
                positive_arg = positive_sample[:, 1]

            elif mode == 'tail-batch':
                positive_arg = positive_sample[:, 2]

            batch_size = positive_sample.size(0)

            for i in range(batch_size):
                # Notice that argsort is not ranking
                ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                assert ranking.size(0) == 1

                ranking = 1 + ranking.item()

                # ranking + 1 is the true ranking used in evaluation metrics
                metrics['MRR'].update(1.0/ranking)

                metrics['MR'].update(ranking)

                metrics['HITS@1'].update(
                    1.0 if ranking <= 1 else 0.0)

                metrics['HITS@3'].update(
                    1.0 if ranking <= 3 else 0.0)

                metrics['HITS@10'].update(
                    1.0 if ranking <= 10 else 0.0)

        return metrics

    @classmethod
    def compute_detailled_score(cls, model, test_set, metrics, types_relations, device):

        for step, (positive_sample, negative_sample, filter_bias, mode) in enumerate(test_set):

            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            filter_bias = filter_bias.to(device)

            score = model(
                sample=positive_sample,
                negative_sample=negative_sample,
                mode=mode,
            )

            score += filter_bias

            argsort = torch.argsort(score, dim=1, descending=True)

            if mode == 'head-batch':
                positive_arg = positive_sample[:, 0]

            elif mode == 'tail-batch':
                positive_arg = positive_sample[:, 2]

            batch_size = positive_sample.size(0)

            for i in range(batch_size):
                # Notice that argsort is not ranking
                ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                assert ranking.size(0) == 1

                ranking = 1 + ranking.item()

                type_relation = types_relations[
                    positive_sample[:, 1][i].item()
                ]

                # ranking + 1 is the true ranking used in evaluation metrics
                metrics[mode][type_relation]['MRR'].update(1.0/ranking)

                metrics[mode][type_relation]['MR'].update(ranking)

                metrics[mode][type_relation]['HITS@1'].update(
                    1.0 if ranking <= 1 else 0.0)

                metrics[mode][type_relation]['HITS@3'].update(
                    1.0 if ranking <= 3 else 0.0)

                metrics[mode][type_relation]['HITS@10'].update(
                    1.0 if ranking <= 10 else 0.0)

        return metrics

    def types_relations(self, model, dataset, threshold):
        """
        Divide input dataset relations into different categories (i.e. ONE-TO-ONE, ONE-TO-MANY,
        MANY-TO-ONE and MANY-TO-MANY) according to the mapping properties of relationships.
        """
        stat_df = pd.DataFrame(self.true_triples)

        stat_df.columns = ['head', 'relation', 'tail']

        mean_head = stat_df[['head', 'relation', 'tail']].groupby(
            ['tail', 'relation']).count().groupby('relation').mean()

        mean_tail = stat_df[['head', 'relation', 'tail']].groupby(
            ['head', 'relation']).count().groupby('relation').mean()

        mean_relations = pd.concat(
            [mean_head, mean_tail], axis='columns').reset_index()

        mean_relations['head'] = mean_relations['head'].apply(
            lambda x: '1' if x <= threshold else 'M')

        mean_relations['tail'] = mean_relations['tail'].apply(
            lambda x: '1' if x <= threshold else 'M')

        mean_relations['type'] = mean_relations['head'] + \
            '_' + mean_relations['tail']

        mapping_type_relations = mean_relations.to_dict()['type']

        return mapping_type_relations

    def detail_eval(self, model, dataset, threshold=1.5):
        """
        Divide input dataset relations into different categories (i.e. ONE-TO-ONE, ONE-TO-MANY,
        MANY-TO-ONE and MANY-TO-MANY) according to the mapping properties of relationships.

        Reference:
            1. [Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." Advances in neural information processing systems. 2013.](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

        """
        mapping_type_relations = self.types_relations(
            model=model,
            dataset=dataset,
            threshold=threshold
        )

        types_relations = ['1_1', '1_M', 'M_1', 'M_M']

        metrics = collections.OrderedDict({
            'head-batch': collections.OrderedDict({}),
            'tail-batch': collections.OrderedDict({})
        })

        for mode in ['head-batch', 'tail-batch']:

            for type_relation in types_relations:

                metrics[mode][type_relation] = collections.OrderedDict({
                    f'{metric}': stats.Mean()
                    for metric in ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']
                })

        with torch.no_grad():

            for test_set in self.get_entity_stream(dataset):

                metrics = self.compute_detailled_score(
                    model=model,
                    test_set=test_set,
                    metrics=metrics,
                    types_relations=mapping_type_relations,
                    device=self.device
                )

        for mode in ['head-batch', 'tail-batch']:
            for type_relation in types_relations:
                for metric in ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']:
                    metrics[mode][type_relation][metric] = round(
                        metrics[mode][type_relation][metric].get(), 4)

        results = pd.DataFrame(metrics)

        head = pd.DataFrame(results['head-batch'].values.tolist())
        tail = pd.DataFrame(results['tail-batch'].values.tolist())

        head.columns = pd.MultiIndex.from_product([["head"], head.columns])
        tail.columns = pd.MultiIndex.from_product([["tail"], tail.columns])
        results = pd.concat([head, tail], axis='columns')
        results = results.set_index(pd.Series(['1_1', '1_M', 'M_1', 'M_M']))
        results.index.name = 'relation'

        # Add frequency of each type of relation:
        frequency = collections.OrderedDict()
        for type_relation in types_relations:
            frequency[type_relation] = 0
        for _, type_relation in mapping_type_relations.items():
            frequency[type_relation] += 1
        for type_relation in types_relations:
            frequency[type_relation] /= len(mapping_type_relations)

        frequency = pd.DataFrame.from_dict(
            frequency,
            orient='index',
            columns=['frequency']
        )

        frequency.columns = pd.MultiIndex.from_product(
            [["metadata"], frequency.columns]
        )

        results = pd.concat([results, frequency], axis='columns')

        return results
