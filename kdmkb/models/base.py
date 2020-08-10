import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BaseModel']


class BaseModel(nn.Module):
    """Knowledge graph embedding base model class.

    Parameters:
        hiddem_dim (int): Embedding size of relations and entities.
        entity_dim (int): Final embedding size of entities.
        relation_dim (int): Final embedding size of relations.
        n_entity (int): Number of entities to consider.
        n_relation (int): Number of relations to consider.
        gamma (float): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.

    Reference:
        1. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, n_entity, n_relation, hidden_dim, entity_dim, relation_dim, gamma):
        super(BaseModel, self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.epsilon = 2

        self.embedding_range = nn.Parameter(
            torch.Tensor(
                [(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(
            torch.zeros(n_entity, self.entity_dim))

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(self.n_relation, self.relation_dim))

        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    @property
    def embeddings(self):
        """Extracts embeddings."""
        entities_embeddings = {}

        for i in range(self.n_entity):
            entities_embeddings[i] = self.entity_embedding[i].detach()

        relations_embeddings = {}

        for i in range(self.n_relation):
            relations_embeddings[i] = self.relation_embedding[i].detach()

        return {'entities': entities_embeddings, 'relations': relations_embeddings}

    @property
    def _repr_title(self):
        return f'{self.__class__.__name__} model'

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.
        This property can be overriden in order to modify the output of the __repr__ method.
        """

        return {
            'Entities embeddings dim': f'{self.entity_dim}',
            'Relations embeddings dim': f'{self.relation_dim}',
            'Gamma': f'{self.gamma.item()}',
            'Number of entities': f'{self.n_entity}',
            'Number of relations': f'{self.n_relation}'
        }

    def __repr__(self):

        l_len = max(map(len, self._repr_content.keys()))
        r_len = max(map(len, self._repr_content.values()))

        return (
            f'{self._repr_title}\n' +
            '\n'.join(
                k.rjust(l_len) + '  ' + v.ljust(r_len)
                for k, v in self._repr_content.items()
            )
        )

    @classmethod
    def format_sample(cls, sample):
        dim_sample = len(sample.shape)

        if dim_sample == 2:
            return sample, (sample.size(0), 1)

        elif dim_sample == 3:
            return sample.view(sample.size(0) * sample.size(1), 3), (sample.size(0), sample.size(1))

    def batch(self, sample):
        sample, shape = self.format_sample(sample=sample)

        head = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=sample[:, 0]
        ).unsqueeze(1)

        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=sample[:, 1]
        ).unsqueeze(1)

        tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=sample[:, 2]
        ).unsqueeze(1)

        return head, relation, tail, shape

    def _set_params(self, entities_embeddings, relations_embeddings, **kwargs):
        """Load pre-trained weights."""
        self.entity_embedding.data.copy_(entities_embeddings)
        self.relation_embedding.data.copy_(relations_embeddings)
        for parameter, weights in kwargs.items():
            self._parameters[parameter].data.copy_(weights)
        return self

    def save(self, path):
        import pickle
        pickle.dump(self, open(path, 'wb'))
