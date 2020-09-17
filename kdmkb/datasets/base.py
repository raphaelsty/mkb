import collections

import numpy as np

import torch

from torch.utils.data import Dataset


__all__ = ['TestDataset', 'TestDatasetRelation', 'TrainDataset']


class TrainDataset(Dataset):
    """Train dataset loader.

    Parameters:
        triples (list): Training set.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        mode (str): head-batch or tail-batch or classification. Mode must be set to tail-batch and
            head-batch when using translationnal models such as RotatE, TransE, DistMult, pRotatE,
            ComplEx. Mode must be set to classification when using ConvE.
        seed (int): Random state.

    Attributes:
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.
        count (dict): Frequency of occurrences of (head, relation) and (relation, tail).
        len (int): Number of training triplets.

    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, triples, entities, relations, mode, pre_compute=True, seed=None):
        self.entities = entities
        self.relations = relations
        self.mode = mode
        self.n_entity = len(self.entities.keys())
        self.n_relation = len(self.relations.keys())
        self.pre_compute = pre_compute
        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

        if self.pre_compute and mode == 'classification':

            self.triples, self.targets = self._pre_compute_classification(
                triples=triples, n_entity=self.n_entity)

        elif self.pre_compute == True and mode != 'classification':

            self.triples, self.weights = self._pre_compute(
                triples=triples)

        elif not self.pre_compute and mode == 'classification':
            self.triples, self.targets = self._light_test_classification(
                triples=triples)
        else:
            self.count = self.get_frequencies(triples)
            self.triples = triples

        self.len = len(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.mode == 'classification' and self.pre_compute:
            return self.triples[idx], self.targets[idx], self.mode

        elif self.pre_compute:
            return self.triples[idx], self.weights[idx], self.mode

        elif self.mode == 'classification' and not self.pre_compute:
            target = torch.zeros(self.n_entity)
            for t in self.targets[idx]:
                target[t] = 1
            return self.triples[idx], target, self.mode

        elif not self.pre_compute:
            h, r, t = self.triples[idx]
            weight = torch.sqrt(
                1 / torch.Tensor([self.count[(h, r)] + self.count[(t, -r-1)]]))
            return torch.LongTensor((h, r, t)), weight, self.mode

    @ staticmethod
    def collate_fn(data):
        """Reshape output data when calling train dataset loader."""
        return (
            torch.stack([_[0] for _ in data], dim=0),
            torch.cat([_[1] for _ in data], dim=0),
            data[0][2]
        )

    @ staticmethod
    def collate_fn_classification(data):
        """Reshape output data when calling train dataset loader."""
        return (
            torch.stack([_[0] for _ in data], dim=0),
            torch.stack([_[1] for _ in data], dim=0),
            data[0][2]
        )

    @staticmethod
    def get_frequencies(triples, start=3):
        count = collections.defaultdict(lambda: start)
        for h, r, t in triples:
            count[(h, r)] += 1
            count[(t, -r-1)] += 1
        return count

    @classmethod
    def _pre_compute(cls, triples, start=3):
        count = cls.get_frequencies(triples=triples)

        train = {}
        weights = {}

        for idx, (h, r, t) in enumerate(triples):
            weights[idx] = torch.sqrt(1 / torch.Tensor(
                [count[(h, r)] + count[(t, -r-1)]]))

            train[idx] = torch.LongTensor((h, r, t))

        return train, weights

    @classmethod
    def _light_test_classification(cls, triples):
        set_head_relation = collections.defaultdict(
            lambda: collections.defaultdict(list))

        for h, r, t in triples:
            set_head_relation[h][r].append(t)

        targets = collections.defaultdict(list)
        train = {}

        idx = 0
        for h, hr in set_head_relation.items():
            for r, hrt in hr.items():
                for t in hrt:
                    train[idx] = torch.LongTensor((h, r))
                    targets[idx].append(t)
                idx += 1
        return train, targets

    @classmethod
    def _pre_compute_classification(cls, triples, n_entity):
        set_head_relation = collections.defaultdict(
            lambda: collections.defaultdict(list))

        for h, r, t in triples:
            set_head_relation[h][r].append(t)

        targets = collections.defaultdict(
            lambda: torch.zeros(n_entity))

        train = {}
        idx = 0
        for h, hr in set_head_relation.items():
            for r, hrt in hr.items():
                for t in hrt:
                    train[idx] = torch.LongTensor((h, r))
                    targets[idx][t] = 1.
                idx += 1

        return train, targets


class TestDataset(Dataset):
    """Test dataset loader dedicated to link prediction.

    Parameters:
        triples (list): Testing set.
        true_triples (list): Triples to filter when validating the model.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        mode (str): head-batch or tail-batch.

    Attributes:
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.
        len (int): Number of training triplets.

    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
    """

    def __init__(self, triples, true_triples, entities, relations, mode):
        self.len = len(triples)
        self.true_triples = set(true_triples)
        self.triples = triples
        self.n_entity = len(entities.keys())
        self.n_relation = len(relations.keys())
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':

            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.true_triples
                   else (-1, head) for rand_head in range(self.n_entity)]
            tmp[head] = (0, head)

        elif self.mode == 'tail-batch':

            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.true_triples
                   else (-1, tail) for rand_tail in range(self.n_entity)]
            tmp[tail] = (0, tail)

        tmp = torch.LongTensor(tmp)

        filter_bias = tmp[:, 0].float()

        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @ staticmethod
    def collate_fn(data):
        """Reshape output data when calling train dataset loader."""
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class TestDatasetRelation(TestDataset):
    """Test dataset loader for relation prediction.

    Parameters:
        triples (list): Testing set.
        true_triples (list): Triples to filter when validating the model.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        mode (str): head-batch or tail-batch.

    Attributes:
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.
        len (int): Number of training triplets.

    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, triples, true_triples, entities, relations):
        super().__init__(
            triples=triples,
            true_triples=true_triples,
            entities=entities,
            relations=relations,
            mode='relation-batch'
        )

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        positive_sample = torch.LongTensor((head, relation, tail))

        tensor_head = torch.tensor([head] * self.n_relation)
        tensor_tail = torch.tensor([tail] * self.n_relation)

        tmp = torch.LongTensor([
            (0, random) if (head, random, tail) not in self.true_triples
            else (-1, relation)
            for random in range(self.n_relation)
        ])

        tensor_relation = tmp[:, 1]
        bias = tmp[:, 0]
        bias[relation] = 0

        negative_sample = torch.stack(
            [tensor_head, tensor_relation, tensor_tail], dim=- 1)

        return positive_sample, negative_sample, bias, self.mode
