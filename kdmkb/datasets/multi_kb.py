from . import Fetch

import numpy as np
import random
import copy


__all__ = ['MultiKb']


class MultiKb(Fetch):
    """Split input dataset into multiples parts and control fraction of aligned entities.

    dataset (kdmkb.datasets): Dataset to split into multiple kg.
    id_set (int): Selected part of the splitted dataset.
    n_part (int): Number of splits of the input dataset.
    aligned_entities (float): Fraction of aligned entities between datasets.

    Example:

        >>> from kdmkb import datasets

        >>> dataset = datasets.MultiKb(
        ...     dataset = datasets.Wn18rr(batch_size = 1, shuffle = True, seed = 42),
        ...     id_set = 0,
        ...     n_part = 2,
        ...     aligned_entities = 0.8,
        ... )

        >>> dataset
            Wn18rr_1_2 dataset
            Batch size          1
            Entities            40923
            Relations           11
            Shuffle             True
            Train triples       43417
            Validation triples  3033
            Test triples        3134
            Wn18rr cutted in    2
            Wn18rr set          1

        >>> for _ in range(3):
        ...     positive_sample, weight, mode = next(dataset)
        ...     print(positive_sample, weight, mode)
        tensor([[6399,    1, 4978]]) tensor([0.3015]) tail-batch
        tensor([[39467,     0, 24602]]) tensor([0.3333]) head-batch
        tensor([[4054,    9, 6759]]) tensor([0.3536]) tail-batch

        >>> assert len(dataset.classification_valid['X']) == len(dataset.classification_valid['y'])
        >>> assert len(dataset.classification_test['X']) == len(dataset.classification_test['y'])

        >>> assert len(dataset.classification_valid['X']) == len(dataset.valid) * 2
        >>> assert len(dataset.classification_test['X']) == len(dataset.test) * 2

    """

    def __init__(self, dataset, id_set, n_part, aligned_entities=1.):

        self.id_set = id_set
        self.n_part = n_part
        self.aligned_entities = aligned_entities
        self.filename = dataset.filename
        self.dataset_name = dataset.name

        super().__init__(
            train=self.split_train(
                train=dataset.train,
                n_part=n_part,
                id_set=id_set,
                seed=dataset.seed
            ),
            valid=dataset.valid,
            test=dataset.test,
            entities=self.corrupt_entities(
                entities=dataset.entities,
                seed=dataset.seed
            ),
            relations=dataset.relations,
            batch_size=dataset.batch_size,
            shuffle=dataset.shuffle,
            num_workers=dataset.num_workers,
            seed=dataset.seed,
            classification_valid=dataset.classification_valid,
            classification_test=dataset.classification_test,
        )

    @property
    def name(self):
        return f'{self.dataset_name}_{self.id_set + 1}_{self.n_part}'

    @property
    def _repr_title(self):
        return f'{self.name} dataset'

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.
        This property can be overriden in order to modify the output of the __repr__ method.
        """
        return {
            'Batch size': f'{self.batch_size}',
            'Entities': f'{self.n_entity}',
            'Relations': f'{self.n_relation}',
            'Shuffle': f'{self.shuffle}',
            'Train triples': f'{len(self.train) if self.train else 0}',
            'Validation triples': f'{len(self.valid) if self.valid else 0}',
            'Test triples': f'{len(self.test) if self.test else 0}',
            f'{self.dataset_name} cutted in': f'{self.n_part}',
            f'{self.dataset_name} set': f'{self.id_set + 1}'
        }

    @classmethod
    def split_train(cls, train, n_part, id_set, seed=42):
        train = copy.deepcopy(train)
        random.Random(seed).shuffle(train)
        return np.array_split(train, n_part)[id_set].tolist()

    def corrupt_entities(self, entities, seed):
        n_entities = len(entities)
        n_entities_to_corrupt = round(n_entities * (1 - self.aligned_entities))
        rng = np.random.RandomState(seed)
        entities_id_corrupt = rng.choice(
            range(n_entities),
            n_entities_to_corrupt,
            replace=False
        )

        e_prime = {value: key for key, value in entities.items()}
        for id_e in entities_id_corrupt:
            e = e_prime[id_e]
            entities.pop(e)
            entities[f'{e}_{self.id_set}_{self.n_part}'] = id_e
        entities = {k: v for k, v in sorted(
            entities.items(), key=lambda item: item[1])}
        return entities
