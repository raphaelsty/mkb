from ..data_loader import fetch_dataset


__all__ = ['WN18RR']


class WN18RR(fetch_dataset.FetchDataset):
    """Iter over WN18RR"""
    def __init__(self, batch_size, negative_sample_size=1024, shuffle=False, num_workers=1):
        triples = [(1, 1, 2), (1, 2, 3)]
        n_entity = 3
        n_relation = 2

        super().__init__(triples=triples, n_entity=n_entity, n_relation=n_relation,
            batch_size=batch_size, negative_sample_size=negative_sample_size, shuffle=shuffle,
            num_workers=num_workers)

    @property
    def n_entity(self):
        return self.n_entity

    @property
    def n_relation(self):
        return self.n_relation
