from .base import BaseModel

__all__ = ["DistMult"]


class DistMult(BaseModel):
    """Dist-Mult model.

    Parameters:
        hiddem_dim (int): Embedding size of relations and entities.
        n_entity (int): Number of entities to consider.
        n_relation (int): Number of relations to consider.
        gamma (float): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.

    Example:

        >>> from mkb import models
        >>> from mkb import datasets

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.CountriesS1(batch_size = 2)

        >>> model = models.DistMult(
        ...    hidden_dim = 3,
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    gamma = 1
        ... )

        >>> model
        DistMult model
            Entities embeddings dim  3
            Relations embeddings dim  3
            Gamma  1.0
            Number of entities  271
            Number of relations  2

        >>> model.embeddings['entities']['oceania']
        tensor([ 0.4845,  0.8654, -0.6108])

        >>> model.embeddings['relations']['locatedin']
        tensor([ 0.3845,  0.5489, -0.2268])

    References:
        1. [Yang, Bishan, et al. "Embedding entities and relations for learning and inference in knowledge bases." arXiv preprint arXiv:1412.6575 (2014).](https://arxiv.org/pdf/1412.6575.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, hidden_dim, entities, relations, gamma):
        super().__init__(
            hidden_dim=hidden_dim,
            relation_dim=hidden_dim,
            entity_dim=hidden_dim,
            entities=entities,
            relations=relations,
            gamma=gamma,
        )

    def forward(self, sample, negative_sample=None, mode=None):
        head, relation, tail, shape = self.batch(
            sample=sample, negative_sample=negative_sample, mode=mode
        )

        if mode == "head-batch":
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)

        return score.view(shape)
