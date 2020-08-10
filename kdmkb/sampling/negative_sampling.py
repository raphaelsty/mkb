import numpy as np

import torch


__all__ = ['NegativeSampling']


class NegativeSampling:
    """Generate negative sample to train models.

    Example:

            >>> from kdmkb import datasets
            >>> from kdmkb import sampling
            >>> from kdmkb import models
            >>> import torch

            >>> _ = torch.manual_seed(42)

            >>> model = models.RotatE(n_entity = 100, n_relation = 100, hidden_dim = 3, gamma = 3)

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

            >>> negative_sample = negative_sampling.generate(positive_sample, mode='tail-batch')

            >>> mode
            'tail-batch'

            >>> positive_sample
            tensor([[0, 0, 1],
                    [1, 0, 2]])

            >>> negative_sample
            tensor([[[0, 0, 2],
                    [0, 0, 3],
                    [0, 0, 0],
                    [0, 0, 2],
                    [0, 0, 2]],
            <BLANKLINE>
                    [[1, 0, 3],
                    [1, 0, 0],
                    [1, 0, 3],
                    [1, 0, 3],
                    [1, 0, 3]]])

            >>> model(negative_sample)
            tensor([[ 1.0213, -3.4006, -3.1048,  1.0213,  1.0213],
                    [-2.4555, -3.9517, -2.4555, -2.4555, -2.4555]], grad_fn=<ViewBackward>)

            Generate fake heads:

            >>> positive_sample, weight, mode = next(dataset)

            >>> negative_sample = negative_sampling.generate(positive_sample, mode)

            >>> mode
            'head-batch'

            >>> positive_sample
            tensor([[0, 0, 1],
                    [1, 0, 2]])

            >>> negative_sample
            tensor([[[1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1]],
            <BLANKLINE>
                    [[0, 0, 2],
                    [0, 0, 2],
                    [3, 0, 2],
                    [0, 0, 2],
                    [3, 0, 2]]])

            >>> model(negative_sample)
            tensor([[-4.7503, -4.7503, -4.7503, -4.7503, -4.7503],
                    [ 1.0213,  1.0213, -0.8139,  1.0213, -0.8139]], grad_fn=<ViewBackward>)

    Reference:
        1. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

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

        self.tensor_head = torch.zeros(self.size)
        self.tensor_relation = torch.zeros(self.size)
        self.tensor_tail = torch.zeros(self.size)

    def _get_sample(self, head, relation, tail, negative_entity, mode):

        if mode == 'head-batch':

            self.tensor_head[:] = negative_entity
            self.tensor_tail[:] = tail

        elif mode == 'tail-batch':

            self.tensor_tail[:] = negative_entity
            self.tensor_head[:] = head

        self.tensor_relation[:] = relation

        return torch.stack([self.tensor_head, self.tensor_relation, self.tensor_tail], dim=-1)

    @classmethod
    def _filter_negative_sample(cls, negative_sample, record):
        mask = np.in1d(
            negative_sample,
            record,
            assume_unique=True,
            invert=True
        )

        return negative_sample[mask]

    def generate(self, positive_sample, mode):
        """Generate negative samples from a head, relation tail

        If the mode is set to head-batch, this method will generate a tensor of fake heads.
        If the mode is set to tail-batch, this method will generate a tensor of fake tails.
        """
        batch_size = positive_sample.shape[0]

        samples = []

        for head, relation, tail in positive_sample:

            head, relation, tail = head.item(), relation.item(), tail.item()

            negative_entity = []

            size = 0

            while size < self.size:

                negative_sample = self._rng.randint(
                    self.n_entity, size=self.size*2)

                if mode == 'head-batch':

                    negative_sample = self._filter_negative_sample(
                        negative_sample=negative_sample,
                        record=self.true_head[(relation, tail)],
                    )

                elif mode == 'tail-batch':

                    negative_sample = self._filter_negative_sample(
                        negative_sample=negative_sample,
                        record=self.true_tail[(head, relation)],
                    )

                negative_entity.append(negative_sample)
                size += negative_sample.size

            negative_entity = np.concatenate(negative_entity)[:self.size]
            negative_entity = torch.LongTensor(negative_entity)

            samples.append(
                self._get_sample(
                    head=head,
                    relation=relation,
                    tail=tail,
                    negative_entity=negative_entity,
                    mode=mode
                )
            )

        samples = torch.stack(samples, dim=0)

        return samples.long()

    @ staticmethod
    def get_true_head_and_tail(triples):
        """Build a dictionary to filter out existing triples from fakes ones."""
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
