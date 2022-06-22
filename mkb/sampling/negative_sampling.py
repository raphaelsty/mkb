import numpy as np
import torch

__all__ = ["NegativeSampling"]


def positive_triples(triples):
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
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))

    for head, relation in true_tail:  # pylint: disable=E1141
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

    return true_head, true_tail


class NegativeSampling:
    """Generate negative sample to train models.

    Example:

            >>> from mkb import datasets
            >>> from mkb import sampling
            >>> from mkb import models
            >>> import torch

            >>> _ = torch.manual_seed(42)

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

            >>> model = models.RotatE(
            ...     entities = entities,
            ...     relations = relations,
            ...     hidden_dim = 3,
            ...     gamma = 3
            ... )

            >>> dataset = datasets.Dataset(
            ...    train = train,
            ...    entities = entities,
            ...    relations = relations,
            ...    batch_size = 2,
            ...    seed = 42,
            ...    shuffle = False,
            ... )

            >>> negative_sampling = sampling.NegativeSampling(
            ...    size = 5,
            ...    train_triples = dataset.train,
            ...    entities = dataset.entities,
            ...    relations = dataset.relations,
            ...    seed = 42,
            ... )

            >>> for data in dataset:
            ...     sample, weight, mode = data['sample'], data['weight'], data['mode']
            ...     break

            >>> negative_sample = negative_sampling.generate(sample, mode='tail-batch')

            >>> mode
            'head-batch'

            >>> sample
            tensor([[0, 0, 1],
                    [1, 0, 2]])

            >>> negative_sample
            tensor([[2, 3, 0, 2, 2],
            [3, 0, 3, 0, 0]])

            >>> model(sample, negative_sample, mode='tail-batch')
            tensor([[-2.7508, -0.8767, -3.1058, -2.7508, -2.7508],
                    [-2.7456, -0.8674, -2.7456, -0.8674, -0.8674]], grad_fn=<ViewBackward>)

            >>> for i, data in enumerate(dataset):
            ...     sample, weight, mode = data['sample'], data['weight'], data['mode']
            ...     if (i + 1) % 2 == 0:
            ...         break

            >>> negative_sample = negative_sampling.generate(sample, mode='head-batch')

            >>> sample
            tensor([[0, 0, 1],
                    [1, 0, 2]])

            >>> negative_sample
            tensor([[2, 2, 2, 2, 2],
            [2, 2, 2, 2, 3]])

            >>> model(sample, negative_sample, mode='head-batch')
            tensor([[-0.3654, -0.3654, -0.3654, -0.3654, -0.3654],
                    [-1.8212, -1.8212, -1.8212, -1.8212, -1.2505]], grad_fn=<ViewBackward>)

    Reference:
        1. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, size, train_triples, entities, relations, seed=42):
        """Generate negative samples.

        size (int): Batch size of the negative samples generated.
        train_triples (list[(int, int, int)]): Set of positive triples.
        entities (dict | list): Set of entities.
        relations (dict | list): Set of relations.
        seed (int): Random state.

        """
        self.size = size

        self.n_entity = len(entities)

        self.n_relation = len(relations)

        self.true_head, self.true_tail = positive_triples(train_triples)

        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

    @classmethod
    def _filter_negative_sample(cls, negative_sample, record):
        mask = np.in1d(negative_sample, record, assume_unique=True, invert=True)
        return negative_sample[mask]

    def generate(self, sample, mode):
        """Generate negative samples from a head, relation tail

        If the mode is set to head-batch, this method will generate a tensor of fake heads.
        If the mode is set to tail-batch, this method will generate a tensor of fake tails.
        """
        samples = []

        negative_entities = self._rng.randint(self.n_entity, size=self.size * 2)

        for head, relation, tail in sample:

            head, relation, tail = head.item(), relation.item(), tail.item()

            negative_entities_sample = []

            size = 0

            while size < self.size:

                if mode == "head-batch":

                    negative_entities_filtered = self._filter_negative_sample(
                        negative_sample=negative_entities,
                        record=self.true_head[(relation, tail)],
                    )

                elif mode == "tail-batch":

                    negative_entities_filtered = self._filter_negative_sample(
                        negative_sample=negative_entities,
                        record=self.true_tail[(head, relation)],
                    )

                size += negative_entities_filtered.size
                negative_entities_sample.append(negative_entities_filtered)

            negative_entities_sample = np.concatenate(negative_entities_sample)[: self.size]

            negative_entities_sample = torch.LongTensor(negative_entities_sample)

            samples.append(negative_entities_sample)

        return torch.stack(samples, dim=0).long()
