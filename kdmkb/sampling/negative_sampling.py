# Reference: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
import numpy as np

import torch


__all__ = ['NegativeSampling']


class NegativeSampling:
    """Generate negative sample to train models.

    Example:

            >>> from kdmkb import datasets
            >>> from kdmkb import sampling

            >>> entities = {
            ...  'e_0': 0,
            ...  'e_1': 1,
            ...  'e_2': 2,
            ...  'e_3': 3,
            ... }

            >>> relations = {
            ...  'r_0': 0,
            ...  'r_1': 1,
            ...  'r_2': 2,
            ...  'r_3': 3,
            ... }

            >>> train = [
            ... (0, 0, 1),
            ... (1, 0, 2),
            ... (2, 0, 3),
            ... (3, 0, 1),
            ... ]

            >>> dataset = datasets.Fetch(
            ...    train = train,
            ...    entities = entities,
            ...    relations = relations,
            ...    batch_size = 2,
            ...    seed = 42
            ... )

            >>> negative_sampling = sampling.NegativeSampling(
            ...    size = 5,
            ...    train_triples = dataset.train,
            ...    entities = dataset.entities,
            ...    relations = dataset.relations,
            ...    seed = 42,
            ... )

            Generate fake tails:
            >>> positive_sample, weight, mode = next(dataset)
            >>> negative_sample = negative_sampling.generate(positive_sample, mode)
            >>> mode
            'tail-batch'
            >>> positive_sample
            tensor([[0, 0, 1],
                    [1, 0, 2]])
            >>> negative_sample
            tensor([[2, 3, 0, 2, 2],
                    [3, 0, 3, 3, 3]])

            Generate fake heads:
            >>> positive_sample, weight, mode = next(dataset)
            >>> negative_sample = negative_sampling.generate(positive_sample, mode)
            >>> mode
            'head-batch'
            >>> positive_sample
            tensor([[0, 0, 1],
                    [1, 0, 2]])
            >>> negative_sample
            tensor([[1, 1, 1, 1, 1],
                    [0, 0, 3, 0, 3]])

    """

    def __init__(self, size, train_triples, entities, relations, seed=42):
        """ Generate negative samples.

            size (int): Batch size of the negative samples generated.
            train_triples (list[(int, int, int)]): Set of positive triples.
            entities (dict | list): Set of entities.
            relations (dict | list): Set of relations.
            seed (int): Random state.

        """
        self.size = size
        self.n_entity = len(entities)
        self.n_relation = len(relations)

        self.true_head, self.true_tail = self.get_true_head_and_tail(
            train_triples)
        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

    def generate(self, positive_sample, mode):
        """Generate negative samples from a head, relation tail

        If the mode is set to head-batch, this method will generate a tensor of fake heads.
        If the mode is set to tail-batch, this method will generate a tensor of fake tails.
        """
        batch_size = positive_sample.shape[0]

        negative_samples = []

        for head, relation, tail in positive_sample:

            head, relation, tail = head.item(), relation.item(), tail.item()

            negative_entity = []

            size = 0

            while size < self.size:

                negative_sample = self._rng.randint(
                    self.n_entity, size=self.size*2)

                if mode == 'head-batch':

                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],
                        assume_unique=True,
                        invert=True
                    )

                elif mode == 'tail-batch':

                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )

                negative_sample = negative_sample[mask]
                negative_entity.append(negative_sample)
                size += negative_sample.size

            negative_entity = np.concatenate(negative_entity)[:self.size]
            negative_entity = torch.from_numpy(negative_entity)

            negative_samples.append(negative_entity)

        negative_samples = torch.cat(
            negative_samples).view(batch_size, self.size)

        return negative_samples

    @staticmethod
    def get_true_head_and_tail(triples):
        """ Build a dictionary to filter out existing triples from fakes ones."""
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:

            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)

            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:  # pylint: disable=E1141
            true_head[(relation, tail)] = np.array(
                list(set(true_head[(relation, tail)])))

        for head, relation in true_tail:  # pylint: disable=E1141
            true_tail[(head, relation)] = np.array(
                list(set(true_tail[(head, relation)])))

        return true_head, true_tail
