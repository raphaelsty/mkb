import torch


__all__ = ['TopK']


class TopK:
    """Retrieve Top heads, relations and tails from input sample and model.

    Parameters:
        entities (dict): Entities of input dataset with label as key and id as value.
        relations (dict): Relations of the input dataset with label as key and id as value.
        device (str): Device to use, cuda or cpu.

    Example:

        >>> from kdmkb import datasets
        >>> from kdmkb import models
        >>> from kdmkb import utils

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.CountriesS1(batch_size = 2, seed = 42)

        >>> model = models.RotatE(
        ...     n_entity = dataset.n_entity,
        ...     n_relation = dataset.n_relation,
        ...     gamma = 3,
        ...     hidden_dim = 4
        ... )

        >>> top_k = utils.TopK(entities = dataset.entities, relations = dataset.relations)

        >>> top_k.top_heads(
        ...     k = 4,
        ...     model = model,
        ...     relation = 0,
        ...     tail = 266,
        ... )
        tensor([197,  50,  75, 176])

        >>> top_k.top_relations(
        ...     k = 1,
        ...     model = model,
        ...     head = 0,
        ...     tail = 266,
        ... )
        tensor([0])

        >>> top_k.top_tails(
        ...     k = 4,
        ...     model = model,
        ...     head = 0,
        ...     relation = 0,
        ... )
        tensor([269, 210, 270, 261])

    """

    def __init__(self, entities, relations, device='cpu'):
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
            model (kdmkb.models): Model.
            relation (int): Id of the relation.
            tail (int): Id of the tail.

        """
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

        return self.entities[rank]

    def top_relations(self, k, model, head, tail):
        """Returns top k relations with a given model, head and tail.

        Parameters:
            k (int): Number of heads to return.
            model (kdmkb.models): Model.
            head (int): Id of the head.
            tail (int): Id of the tail.

        """
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

        return self.relations[rank]

    def top_tails(self, k, model, head, relation):
        """Returns top k tails with a given model, head and relation.

        Parameters:
            k (int): Number of heads to return.
            model (kdmkb.models): Model.
            head (int): Id of the head.
            relation (int): Id of the relation.

        """
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

        return self.entities[rank]

    @classmethod
    def _get_rank(cls, model, sample, k, device):
        with torch.no_grad():
            return torch.argsort(
                model(sample.to(device)),
                descending=True,
                dim=0
            ).flatten()[:k]
