import torch
import torch.nn as nn

from ..text.scoring import RotatE as TextRotatE

__all__ = ["BaseModel", "BaseConvE", "TextBaseModel"]


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def _repr_title(self):
        return f"{self.name} model"

    def __repr__(self):
        l_len = max(map(len, self._repr_content.keys()))
        r_len = max(map(len, self._repr_content.values()))

        return f"{self._repr_title}\n" + "\n".join(
            k.rjust(l_len) + "  " + v.ljust(r_len) for k, v in self._repr_content.items()
        )

    def save(self, path):
        import pickle

        with open(path, "wb") as handle:
            pickle.dump(self.cpu().eval(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def forward(self):
        pass

    def distill(self):
        pass

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.
        This property can be overriden in order to modify the output of the __repr__ method.
        """
        return {}


class BaseModel(Base):
    """Knowledge graph embedding base model class.

    Parameters:
        hiddem_dim (int): Embedding size of relations and entities.
        entity_dim (int): Final embedding size of entities.
        relation_dim (int): Final embedding size of relations.
        entities (dict): Mapping between entities and ids.
        relations (dict): Mapping between relations and ids.
        gamma (float): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.

    Reference:
        1. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, entities, relations, hidden_dim, entity_dim, relation_dim, gamma):
        super().__init__()

        self.entities = {i: e for e, i in entities.items()}
        self.relations = {i: r for r, i in relations.items()}
        self.n_entity = len(entities)
        self.n_relation = len(relations)
        self.hidden_dim = hidden_dim
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.epsilon = 2

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False,
        )

        self.entity_embedding = nn.Parameter(torch.zeros(self.n_entity, self.entity_dim))

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.relation_embedding = nn.Parameter(torch.zeros(self.n_relation, self.relation_dim))

        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

    @property
    def embeddings(self):
        """Extracts embeddings."""
        entities_embeddings = {}

        for i in range(self.n_entity):
            entities_embeddings[self.entities[i]] = self.entity_embedding[i].detach()

        relations_embeddings = {}

        for i in range(self.n_relation):
            relations_embeddings[self.relations[i]] = self.relation_embedding[i].detach()

        return {"entities": entities_embeddings, "relations": relations_embeddings}

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.
        This property can be overriden in order to modify the output of the __repr__ method.
        """

        return {
            "Entities embeddings dim": f"{self.entity_dim}",
            "Relations embeddings dim": f"{self.relation_dim}",
            "Gamma": f"{self.gamma.item()}",
            "Number of entities": f"{self.n_entity}",
            "Number of relations": f"{self.n_relation}",
        }

    @staticmethod
    def format_sample(sample, negative_sample=None):
        """Adapt input tensor to compute scores."""
        dim_sample = len(sample.shape)

        if dim_sample == 2:

            if negative_sample is None:

                return sample, (sample.size(0), 1)

            else:

                return sample, negative_sample.shape

        elif dim_sample == 3:

            return (
                sample.view(sample.size(0) * sample.size(1), 3),
                (sample.size(0), sample.size(1)),
            )

    def batch(self, sample, negative_sample=None, mode=None):
        sample, shape = self.format_sample(sample=sample, negative_sample=negative_sample)

        if mode == "head-batch":
            head, relation, tail = self.head_batch(sample=sample, negative_sample=negative_sample)

        elif mode == "tail-batch":
            head, relation, tail = self.tail_batch(sample=sample, negative_sample=negative_sample)
        else:
            head, relation, tail = self.default_batch(sample=sample)

        return head, relation, tail, shape

    def default_batch(self, sample):
        head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)

        relation = torch.index_select(
            self.relation_embedding, dim=0, index=sample[:, 1]
        ).unsqueeze(1)

        tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

        return head, relation, tail

    def head_batch(self, sample, negative_sample):
        """Used to get faster when computing scores for negative samples."""
        batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)

        head = torch.index_select(
            self.entity_embedding, dim=0, index=negative_sample.view(-1)
        ).view(batch_size, negative_sample_size, -1)

        relation = torch.index_select(
            self.relation_embedding, dim=0, index=sample[:, 1]
        ).unsqueeze(1)

        tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

        return head, relation, tail

    def tail_batch(self, sample, negative_sample):
        """Used to get faster when computing scores for negative samples."""
        batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)

        head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)

        relation = torch.index_select(
            self.relation_embedding, dim=0, index=sample[:, 1]
        ).unsqueeze(1)

        tail = torch.index_select(
            self.entity_embedding, dim=0, index=negative_sample.view(-1)
        ).view(batch_size, negative_sample_size, -1)

        return head, relation, tail

    def _set_params(self, entities_embeddings, relations_embeddings, **kwargs):
        """Load pre-trained weights."""
        self.entity_embedding.data.copy_(entities_embeddings)
        self.relation_embedding.data.copy_(relations_embeddings)
        for parameter, weights in kwargs.items():
            self._parameters[parameter].data.copy_(weights)
        return self

    def distill(self, sample, negative_sample=None, mode=None):
        """Default distillation method"""
        return self(sample=sample, negative_sample=negative_sample, mode=mode)


def mean_pooling(hidden_state, attention_mask):
    """Mean pooling.

    References
    ----------
    1. [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class TextBaseModel(BaseModel):
    """Textual base model class.

    Examples
    --------

    >>> from mkb import datasets, models, text
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> dataset = datasets.Semanlink(1, pre_compute=False)

    >>> model = models.TextBaseModel(
    ...     entities = dataset.entities,
    ...     relations=dataset.relations,
    ...     hidden_dim=3,
    ...     gamma=3,
    ...     scoring=text.TransE(),
    ... )

    >>> sample = torch.tensor([[3, 0, 4], [5, 1, 6]])

    >>> head, relation, tail, shape = model.batch(sample)

    >>> head
    ['Théorie des cordes', 'JCS - Java Caching System']

    >>> tail
    ['rdfQuery', 'Over-Engineering']

    """

    def __init__(self, entities, relations, hidden_dim, scoring, gamma):

        relation_dim = hidden_dim
        entity_dim = hidden_dim

        if isinstance(scoring, TextRotatE):
            relation_dim = relation_dim // 2

        super().__init__(
            entities=entities,
            relations=relations,
            hidden_dim=hidden_dim,
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            gamma=gamma,
        )

        self.scoring = scoring

        self.entities = {i: e for e, i in entities.items()}
        self.relations = {i: r for r, i in relations.items()}

        self.n_entity = len(entities)
        self.n_relation = len(relations)
        self.hidden_dim = hidden_dim
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + 2) / self.hidden_dim]),
            requires_grad=False,
        )

        self.relation_embedding = nn.Parameter(torch.zeros(self.n_relation, self.relation_dim))

        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

    @property
    def twin(self):
        return False

    def forward(self, sample, negative_sample=None, mode=None):
        """Compute scores of input sample, negative sample with respect to the mode."""

        head, relation, tail, shape = self.encode(
            sample=sample, negative_sample=negative_sample, mode=mode
        )

        score = self.scoring(
            **{
                "head": head,
                "relation": relation,
                "tail": tail,
                "gamma": self.gamma,
                "mode": mode,
                "embedding_range": self.embedding_range,
                "modulus": self.modulus,
            }
        )

        return score.view(shape)

    def encode(self, sample, negative_sample=None, mode=None):
        """Encode input sample, negative sample with respect to the mode."""

        head, relation, tail, shape = self.batch(
            sample=sample, negative_sample=negative_sample, mode=mode
        )

        if negative_sample is None:

            head = self.encoder(e=head, mode="head").unsqueeze(1)
            tail = self.encoder(e=tail, mode="tail").unsqueeze(1)

        else:

            head, tail = self.negative_encoding(
                sample=sample,
                head=head,
                tail=tail,
                negative_sample=negative_sample,
                mode=mode,
            )

        return head, relation, tail, shape

    def batch(self, sample, negative_sample=None, mode=None):
        """Process input sample."""
        sample, shape = self.format_sample(sample=sample, negative_sample=negative_sample)

        relation = torch.index_select(
            self.relation_embedding, dim=0, index=sample[:, 1]
        ).unsqueeze(1)

        head = sample[:, 0]
        tail = sample[:, 2]

        head = [self.entities[h.item()] for h in head]
        tail = [self.entities[t.item()] for t in tail]

        return head, relation, tail, shape

    def negative_encoding(self, sample, head, tail, negative_sample, mode):

        mode_encoder = "head" if mode == "head-batch" else "tail"

        negative_sample = torch.stack(
            [
                self.encoder([self.entities[e.item()] for e in ns], mode=mode_encoder)
                for ns in negative_sample
            ]
        )

        if mode == "head-batch":

            head = negative_sample

            tail = self.encoder(e=tail, mode="tail").unsqueeze(1)

        elif mode == "tail-batch":

            tail = negative_sample

            head = self.encoder(e=head, mode="head").unsqueeze(1)

        return head, tail

    def encoder(self, e):
        """Encoder should be defined in the children class."""
        return torch.zeros(len(e))
