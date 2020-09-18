import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BaseModel', 'BaseConvE']


class Base(nn.Module):

    def __init__(self):
        super(Base, self).__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def _repr_title(self):
        return f'{self.name} model'

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

    def save(self, path):
        import pickle

        with open(path, 'wb') as handle:
            pickle.dump(
                self.cpu().eval(),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )

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
        n_entity (int): Number of entities to consider.
        n_relation (int): Number of relations to consider.
        gamma (float): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.

    Reference:
        1. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, n_entity, n_relation, hidden_dim, entity_dim, relation_dim, gamma):
        super().__init__()

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
                (sample.size(0), sample.size(1))
            )

    def batch(self, sample, negative_sample=None, mode=None):
        sample, shape = self.format_sample(
            sample=sample,
            negative_sample=negative_sample
        )

        if mode == 'head-batch':
            head, relation, tail = self.head_batch(
                sample=sample,
                negative_sample=negative_sample
            )

        elif mode == 'tail-batch':
            head, relation, tail = self.tail_batch(
                sample=sample,
                negative_sample=negative_sample
            )
        else:
            head, relation, tail = self.default_batch(sample=sample)

        return head, relation, tail, shape

    def default_batch(self, sample):
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

        return head, relation, tail

    def head_batch(self, sample, negative_sample):
        """Used to get faster when computing scores for negative samples."""
        batch_size, negative_sample_size = negative_sample.size(
            0), negative_sample.size(1)

        head = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=negative_sample.view(-1)
        ).view(batch_size, negative_sample_size, -1)

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

        return head, relation, tail

    def tail_batch(self, sample, negative_sample):
        """Used to get faster when computing scores for negative samples."""
        batch_size, negative_sample_size = negative_sample.size(
            0), negative_sample.size(1)

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
            index=negative_sample.view(-1)
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


class BaseConvE(Base):
    """ConvE base model class.

    Parameters:
        hiddem_dim (int): Embedding size of relations and entities.
        entity_dim (int): Final embedding size of entities.
        relation_dim (int): Final embedding size of relations.
        n_entity (int): Number of entities to consider.
        n_relation (int): Number of relations to consider.
        gamma (float): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.

    """

    def __init__(
        self,
        n_entity,
        n_relation,
        hidden_dim_w,
        hidden_dim_h,
        channels,
        kernel_size,
        embedding_dropout,
        feature_map_dropout,
        layer_dropout,
    ):
        super().__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim_w = hidden_dim_w
        self.hidden_dim_h = hidden_dim_h
        self.channels = channels
        self.kernel_size = kernel_size
        self.embedding_dropout = embedding_dropout
        self.feature_map_dropout = feature_map_dropout
        self.layer_dropout = layer_dropout

        self.hidden_dim = hidden_dim_h * hidden_dim_w

        self.flattened_size = (
            (self.hidden_dim_w * 2 - self.kernel_size + 1) *
            (self.hidden_dim_h - self.kernel_size + 1) * self.channels
        )

        self.entity_embedding = nn.Embedding(
            self.n_entity,
            self.hidden_dim,
            padding_idx=0
        )

        self.relation_embedding = nn.Embedding(
            self.n_relation,
            self.hidden_dim,
            padding_idx=0
        )

    @property
    def embeddings(self):
        """Extracts embeddings."""
        entities_embeddings = {}

        for i in range(self.n_entity):
            entities_embeddings[i] = self.entity_embedding(
                torch.tensor([[i]])
            ).flatten().detach()

        relations_embeddings = {}

        for i in range(self.n_relation):
            relations_embeddings[i] = self.relation_embedding(
                torch.tensor([[i]])
            ).flatten().detach()

        return {'entities': entities_embeddings, 'relations': relations_embeddings}

    @ property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.
        This property can be overriden in order to modify the output of the __repr__ method.
        """

        return {
            'Entities embeddings dim': f'{self.hidden_dim}',
            'Relations embeddings dim': f'{self.hidden_dim}',
            'Number of entities': f'{self.n_entity}',
            'Number of relations': f'{self.n_relation}',
            'Channels': f'{self.channels}',
            'Kernel size': f'{self.kernel_size}',
            'Embeddings dropout': f'{self.embedding_dropout}',
            'Feature map dropout': f'{self.feature_map_dropout}',
            'Layer dropout': f'{self.layer_dropout}'
        }

    def format_sample(self, sample, negative_sample=None):
        """Adapt input tensor to compute scores."""
        dim_sample = len(sample.shape)

        if dim_sample == 2:
            # Classification mode, output a probability distribution.
            if sample.shape[1] == 2 and negative_sample is None:
                return sample, (sample.size(0), self.n_entity)

            # Default mode, compute score for input samples.
            elif negative_sample is None:
                return sample, (sample.size(0), 1)

            # head-batch or tail-batch mode.
            else:
                return sample, negative_sample.shape

        # Distillation mode.
        elif dim_sample == 3:

            return (
                sample.view(sample.size(0) * sample.size(1), 3),
                (sample.size(0), sample.size(1))
            )

    def batch(self, sample, negative_sample, mode):
        sample, shape = self.format_sample(
            sample=sample,
            negative_sample=negative_sample,
        )

        if mode == 'classification':
            head, relation = self.classification_batch(sample=sample)
            tail = None

        elif mode == 'head-batch':
            head, relation, tail = self.head_batch(
                sample=sample, negative_sample=negative_sample)

        elif mode == 'tail-batch':
            head, relation, tail = self.tail_batch(
                sample=sample, negative_sample=negative_sample)

        elif mode == 'default':
            head, relation, tail = self.default_batch(sample=sample)

        return head, relation, tail, shape

    def head_batch(self, sample, negative_sample):
        head = self.entity_embedding(negative_sample)
        relation = self.relation_embedding(sample[:, 1])
        tail = self.entity_embedding(sample[:, 2])

        relation = torch.stack(
            [relation for _ in range(negative_sample.shape[1])], dim=1)

        tail = torch.stack(
            [tail for _ in range(negative_sample.shape[1])], dim=1)
        tail = tail.view(tail.shape[0] * tail.shape[1], tail.shape[2], 1)

        return head, relation, tail

    def tail_batch(self, sample, negative_sample):
        head = self.entity_embedding(sample[:, 0])
        relation = self.relation_embedding(sample[:, 1])
        tail = self.entity_embedding(negative_sample).transpose(1, 2)
        return head, relation, tail

    def default_batch(self, sample):
        head = self.entity_embedding(sample[:, 0])
        relation = self.relation_embedding(sample[:, 1])
        tail = self.entity_embedding(sample[:, 2]).unsqueeze(-1)
        return head, relation, tail

    def classification_batch(self, sample):
        head = self.entity_embedding(sample[:, 0])
        relation = self.relation_embedding(sample[:, 1])
        return head, relation
