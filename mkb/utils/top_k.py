import torch


__all__ = ['TopK']


class TopK:
    """Retrieve Top heads, relations and tails from input sample and model.

    Parameters:
        entities (dict): Entities of input dataset with label as key and id as value.
        relations (dict): Relations of the input dataset with label as key and id as value.
        device (str): Device to use, cuda or cpu.

    Example:

        >>> from mkb import datasets
        >>> from mkb import models
        >>> from mkb import utils

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.CountriesS1(batch_size = 2, seed = 42)

        >>> model = models.RotatE(
        ...     entities = dataset.entities,
        ...     relations = dataset.relations,
        ...     gamma = 3,
        ...     hidden_dim = 4
        ... )

        >>> top_k = utils.TopK(entities = dataset.entities, relations = dataset.relations)

        >>> top_k.top_heads(
        ...     k = 4,
        ...     model = model,
        ...     relation = 'neighbor',
        ...     tail = 'western_africa',
        ... )
        ['mauritius', 'são_tomé_and_príncipe', 'guinea-bissau', 'saint_kitts_and_nevis']

        >>> top_k.top_relations(
        ...     k = 4,
        ...     model = model,
        ...     head = 'azerbaijan',
        ...     tail = 'western_africa',
        ... )
        ['locatedin', 'neighbor']

        >>> top_k.top_tails(
        ...     k = 4,
        ...     model = model,
        ...     head = 'western_africa',
        ...     relation = 'neighbor',
        ... )
        ['afghanistan', 'barbados', 'taiwan', 'new_caledonia']

    """

    def __init__(self, entities, relations, device='cpu'):
        self.mapping_entities = entities
        self.mapping_relations = relations

        self.reverse_mapping_entities = {
            value: key for key, value in self.mapping_entities.items()}

        self.reverse_mapping_relations = {
            value: key for key, value in self.mapping_relations.items()}

        self.entities = torch.tensor(
            [e for _, e in entities.items()],
            dtype=int
        )

        self.relations = torch.tensor(
            [r for _, r in relations.items()],
            dtype=int
        )

        self.device = device

        self.default_heads_e = torch.zeros(self.entities.shape[0], dtype=int)
        self.default_relations_e = torch.zeros(
            self.entities.shape[0], dtype=int)
        self.default_tails_e = torch.zeros(self.entities.shape[0], dtype=int)

        self.default_heads_r = torch.zeros(self.relations.shape[0], dtype=int)
        self.default_tails_r = torch.zeros(self.relations.shape[0], dtype=int)

    def top_heads(self, k, model, relation, tail):
        """Returns top k heads with a given model, relation and tail.

        Parameters:
            k (int): Number of heads to return.
            model (mkb.models): Model.
            relation (int | str): Id | label of the relation.
            tail (int | str): Id | label of the tail.

        """
        if isinstance(relation, str):
            relation = self.mapping_relations[relation]

        if isinstance(tail, str):
            tail = self.mapping_entities[tail]

        training = False
        if model.training:
            training = True
            model = model.eval()

        self.default_relations_e[:] = relation
        self.default_tails_e[:] = tail

        tensor_heads = torch.stack(
            [
                self.entities,
                self.default_relations_e,
                self.default_tails_e,
            ],
            dim=1
        )

        rank = self._get_rank(
            model=model,
            sample=tensor_heads,
            k=k,
            device=self.device
        )

        if training:
            model = model.train()

        return [self.reverse_mapping_entities[e.item()] for e in self.entities[rank]]

    def top_relations(self, k, model, head, tail):
        """Returns top k relations with a given model, head and tail.

        Parameters:
            k (int): Number of heads to return.
            model (mkb.models): Model.
            head (int | str): Id | label of the head.
            tail (int | str): Id | label of the tail.

        """
        if isinstance(head, str):
            head = self.mapping_entities[head]

        if isinstance(tail, str):
            tail = self.mapping_entities[tail]

        training = False
        if model.training:
            training = True
            model = model.eval()

        self.default_heads_r[:] = head
        self.default_tails_r[:] = tail

        tensor_relations = torch.stack(
            [
                self.default_heads_r,
                self.relations,
                self.default_tails_r,
            ],
            dim=1
        )

        rank = self._get_rank(
            model=model,
            sample=tensor_relations,
            k=k,
            device=self.device
        )

        if training:
            model = model.train()

        return [self.reverse_mapping_relations[r.item()] for r in self.relations[rank]]

    def top_tails(self, k, model, head, relation):
        """Returns top k tails with a given model, head and relation.

        Parameters:
            k (int): Number of heads to return.
            model (mkb.models): Model.
            head (int | str): Id | label of the head.
            relation (int | str): Id | label of the relation.

        """
        if isinstance(head, str):
            head = self.mapping_entities[head]

        if isinstance(relation, str):
            relation = self.mapping_relations[relation]

        training = False
        if model.training:
            training = True
            model = model.eval()

        self.default_heads_e[:] = head
        self.default_relations_e[:] = relation

        tensor_tails = torch.stack(
            [
                self.default_heads_e,
                self.default_relations_e,
                self.entities,
            ],
            dim=1
        )

        rank = self._get_rank(
            model=model,
            sample=tensor_tails,
            k=k,
            device=self.device
        )

        if training:
            model = model.train()

        return [self.reverse_mapping_entities[e.item()] for e in self.entities[rank]]

    @classmethod
    def _get_rank(cls, model, sample, k, device):
        with torch.no_grad():
            return torch.argsort(
                model(sample.to(device)),
                descending=True,
                dim=0
            ).flatten()[:k]
