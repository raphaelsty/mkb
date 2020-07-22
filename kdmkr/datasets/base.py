# Reference: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
import numpy as np

import torch

from torch.utils.data import Dataset


__all__ = ['TestDataset', 'TrainDataset']


class TrainDataset(Dataset):
    """Loader for training set."""
    def __init__(self, triples, entities, relations, mode, seed=42):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.entities = entities
        self.relations = relations
        self.mode = mode
        self.seed = seed
        self.n_entity = len(self.entities.keys())
        self.n_relation = len(self.relations.keys())
        self.count = self.count_frequency(triples)
        self._rng = np.random.RandomState(self.seed) # pylint: disable=no-member

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        subsample_weight = torch.cat([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, entities, relations, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.n_entity = len(entities.keys())
        self.n_relation = len(relations.keys())
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.n_entity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.n_entity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode



