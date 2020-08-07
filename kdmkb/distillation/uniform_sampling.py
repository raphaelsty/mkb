import torch

import numpy as np

import collections

__all__ = ['UniformSampling']


class UniformSampling:
    """Supervised uniform sampling dedicated to distillation.

    Creates 3 tensors for the student and the teacher for each single training sample. Those tensors
    are made of indexes and allows to computes distribution probability on a subset of entities and
    relations of the knowledge graph. Uniform sampling is supervised, it includes the ground truth
    in the probability distribution.

    Parameters:
        batch_size_entity (int): Number of entities to consider to compute distribution probability
            when using distillation.
        batch_size_relation (int): Number of relations to consider to compute distribution
            probability when using distillation.
        seed (int): Random state.

    Example:

        >>> from kdmkb import distillation

        >>> teacher_entities = {'e_1': 0, 'e_2': 1, 'e_3': 2, 'e_4': 3, 'e_5': 4}
        >>> teacher_relations = {'r_1': 0, 'r_2': 1, 'r_3': 2, 'r_4': 3, 'r_5': 4}

        >>> student_entities = {'e_2': 0, 'e_3': 1, 'e_4': 2, 'e_5': 3}
        >>> student_relations = {'r_2': 0, 'r_3': 1, 'r_4': 2, 'r_5': 3}

        >>> mapping_entities = collections.OrderedDict({
        ...    i: student_entities[e] for e, i in teacher_entities.items()
        ...    if e in student_entities
        ... })

        >>> mapping_relations = collections.OrderedDict({
        ...    i: student_relations[e] for e, i in teacher_relations.items()
        ...    if e in student_relations
        ... })

        >>> uniform_sampling = distillation.UniformSampling(
        ...    batch_size_entity   = 4,
        ...    batch_size_relation = 4,
        ...    seed                = 43,
        ... )

        >>> ( head_distribution_teacher, relation_distribution_teacher,
        ...    tail_distribution_teacher, head_distribution_student,
        ...    relation_distribution_student, tail_distribution_student,
        ... ) = uniform_sampling.get(
        ...     positive_sample_size = 1,
        ...     mapping_entities     = mapping_entities,
        ...     mapping_relations    = mapping_relations,
        ... )

        >>> head_distribution_teacher
        tensor([[3., 2., 4., 1.]])

        >>> relation_distribution_teacher
        tensor([[3., 1., 4., 2.]])

        >>> tail_distribution_teacher
        tensor([[3., 2., 4., 1.]])

        >>> head_distribution_student
        tensor([[2., 1., 3., 0.]])

        >>> relation_distribution_student
        tensor([[2., 0., 3., 1.]])

        >>> tail_distribution_student
        tensor([[2., 1., 3., 0.]])

    """

    def __init__(self, batch_size_entity, batch_size_relation, seed=None, **kwargs):
        self.batch_size_entity = batch_size_entity
        self.batch_size_relation = batch_size_relation
        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

    @property
    def supervised(self):
        """Include the ground truth."""
        return True

    def get(self, mapping_entities, mapping_relations, positive_sample_size, **kwargs):
        """
        """
        entity_distribution_teacher = self._rng.choice(
            a=list(mapping_entities.keys()), size=self.batch_size_entity, replace=False)

        relation_distribution_teacher = self._rng.choice(
            a=list(mapping_relations.keys()),  size=self.batch_size_relation, replace=False)

        entity_distribution_student = [
            mapping_entities[entity] for entity in entity_distribution_teacher]

        relation_distribution_student = [
            mapping_relations[relation] for relation in relation_distribution_teacher]

        entity_distribution_teacher = torch.Tensor(entity_distribution_teacher).view(
            1, self.batch_size_entity)

        entity_distribution_student = torch.Tensor(entity_distribution_student).view(
            1, self.batch_size_entity)

        relation_distribution_teacher = torch.Tensor(relation_distribution_teacher).view(
            1, self.batch_size_relation)

        relation_distribution_student = torch.Tensor(relation_distribution_student).view(
            1, self.batch_size_relation)

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
