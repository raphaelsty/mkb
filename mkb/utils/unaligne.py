import secrets

__all__ = ["Unaligne"]


class Unaligne:
    """
    Unaligne entities and relations.

    Parameters:
        rate (float): Rate of entities and relations to unaligne.
        unaligne_entities (bool): Wether or note unaligne entities.
        unaligne_relations (bool): Wether or note unaligne relations.

    Example:

        >>> from mkb import datasets
        >>> from mkb import utils

        >>> dataset = datasets.CountriesS1(1)
        >>> dataset = utils.Unaligne(rate = 0.2)(dataset)

        >>> dataset.relations
        {'locatedin': 0, 'neighbor': 1}

    """

    def __init__(self, rate, unaligne_entities=True, unaligne_relations=True):
        self.rate = rate
        self.unaligne_entities = unaligne_entities
        self.unaligne_relations = unaligne_relations

    def __call__(self, dataset):
        if self.unaligne_entities:
            dataset.entities = self.process(dataset.entities)
        if self.unaligne_relations:
            dataset.relations = self.process(dataset.relations)
        return dataset

    def process(self, X):
        threshold = len(X) * self.rate // 1
        for i, x in enumerate(X):
            if i >= threshold:
                return dict(sorted(X.items(), key=lambda item: item[1]))
            else:
                X[f"{x}_{secrets.token_hex(nbytes=3)}"] = X.pop(x)
        return dict(sorted(X.items(), key=lambda item: item[1]))
