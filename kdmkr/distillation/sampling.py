import torch

import numpy as np

__all__ = ['UniformSampling', 'TopkSampling']

class UniformSampling:
    """Init tensor dedicated to distillation with uniform sampling for the student and
    the teacher. The sampling method must returns 6 tensors dedicated to the distribution
    of heads, relations and tails for both teacher and student. Each tensor must have the shape
    (positive_sample_size, 3).

    Example:

        >>> from kdmkr import distillation

        >>> mapping_entities = {1: 0, 2: 1, 3: 2}
        >>> mapping_relations = {1: 0, 2: 1, 3: 2}

        >>> uniform_sampling = distillation.UniformSampling()

        >>> (
        ...    head_distribution_teacher, relation_distribution_teacher,
        ...    tail_distribution_teacher, head_distribution_student,
        ...    relation_distribution_student, tail_distribution_student,
        ... ) = uniform_sampling(
        ...    mapping_entities     = mapping_entities,
        ...    mapping_relations    = mapping_relations,
        ...    batch_size_entity    = 3,
        ...    batch_size_relation  = 3,
        ...    positive_sample_size = 1,
        ...    seed                 = 42,
        ... )

        >>> head_distribution_teacher
        tensor([[1, 2, 3]])

        >>> relation_distribution_teacher
        tensor([[2, 3, 1]])

        >>> tail_distribution_teacher
        tensor([[1, 2, 3]])

        >>> head_distribution_student
        tensor([[0, 1, 2]])

        >>> relation_distribution_student
        tensor([[1, 2, 0]])

        >>> tail_distribution_student
        tensor([[0, 1, 2]])

    """
    def __init__(self):
        pass

    @property
    def supervised(self):
        """Distillation module will include the ground-truth in the sample if the sampler is
        supervised.
        """
        return True

    def __call__(self, mapping_entities, mapping_relations, batch_size_entity, batch_size_relation,
        positive_sample_size, seed, **kwargs):
        """
        """
        rng = np.random.seed(seed)

        entity_distribution_teacher = np.random.choice(
            a=list(mapping_entities.keys()), size=batch_size_entity, replace=False,
        )

        relation_distribution_teacher = np.random.choice(
            a=list(mapping_relations.keys()),  size=batch_size_relation, replace=False,
        )

        entity_distribution_student = [
            mapping_entities[entity] for entity in entity_distribution_teacher]

        relation_distribution_student = [
            mapping_relations[relation] for relation in relation_distribution_teacher]

        entity_distribution_teacher = torch.tensor(entity_distribution_teacher).view( # pylint: disable=not-callable
            1, batch_size_entity)

        entity_distribution_student = torch.tensor(entity_distribution_student).view( # pylint: disable=not-callable
            1, batch_size_entity)

        relation_distribution_teacher = torch.tensor(relation_distribution_teacher).view( # pylint: disable=not-callable
            1, batch_size_relation)

        relation_distribution_student = torch.tensor(relation_distribution_student).view( # pylint: disable=not-callable
            1, batch_size_relation)

        head_distribution_teacher = torch.cat(
            positive_sample_size * [entity_distribution_teacher])

        relation_distribution_teacher = torch.cat(
            positive_sample_size * [relation_distribution_teacher])

        tail_distribution_teacher = torch.cat(
            positive_sample_size * [entity_distribution_teacher])

        head_distribution_student = torch.cat(
            positive_sample_size * [entity_distribution_student])

        relation_distribution_student = torch.cat(
            positive_sample_size * [relation_distribution_student])

        tail_distribution_student = torch.cat(
            positive_sample_size * [entity_distribution_student])

        return (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
            head_distribution_student, relation_distribution_student, tail_distribution_student)


class TopkSampling:

    def __init__(self):
        pass

    def __call__(self, **kwargs):
        pass
