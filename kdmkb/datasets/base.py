import collections

import numpy as np

import torch

from torch.utils.data import Dataset


__all__ = ['TestDataset', 'TrainDataset']


class TrainDataset(Dataset):
    """Train dataset loader.

    Parameters:
        triples (list): Training set.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        mode (str): head-batch or tail-batch.
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

    def __init__(self, triples, entities, relations, mode, seed=None):
        self.len = len(triples)
        self.triples = triples
        self.entities = entities
        self.relations = relations
        self.n_entity = len(self.entities.keys())
        self.n_relation = len(self.relations.keys())
        self.mode = mode
        self.count = self.count_frequency(triples)
        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(
            head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        """Reshape output data when calling train dataset loader."""
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        subsample_weight = torch.cat([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=3):
        count = collections.defaultdict(lambda: start)
        for head, relation, tail in triples:
            count[(head, relation)] += 1
            count[(tail, -relation-1)] += 1
        return count


class TestDataset(Dataset):
    """Test dataset loader.

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

        else:
            raise ValueError(
                'negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        """Reshape output data when calling train dataset loader."""
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode