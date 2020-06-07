import torch

import numpy as np

import collections

import faiss

__all__ = ['UniformSampling', 'TopKSampling']

class UniformSampling:
    """Init tensor dedicated to distillation with uniform sampling for the student and
    the teacher. The sampling method must returns 6 tensors dedicated to the distribution
    of heads, relations and tails for both teacher and student. Each tensor must have the shape
    (positive_sample_size, 3).

    Example:

        >>> from kdmkr import distillation

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
        ...    batch_size_entity    = 4,
        ...    batch_size_relation  = 4,
        ...    seed                 = 43,
        ... )

        >>> ( head_distribution_teacher, relation_distribution_teacher,
        ...    tail_distribution_teacher, head_distribution_student,
        ...    relation_distribution_student, tail_distribution_student,
        ... ) = uniform_sampling.get(
        ...     mapping_entities = mapping_entities,
        ...     mapping_relations = mapping_relations,
        ...     positive_sample_size = 1
        ... )

        >>> head_distribution_teacher
        tensor([[3, 2, 4, 1]])

        >>> relation_distribution_teacher
        tensor([[3, 1, 4, 2]])

        >>> tail_distribution_teacher
        tensor([[3, 2, 4, 1]])

        >>> head_distribution_student
        tensor([[2, 1, 3, 0]])

        >>> relation_distribution_student
        tensor([[2, 0, 3, 1]])

        >>> tail_distribution_student
        tensor([[2, 1, 3, 0]])

    """
    def __init__(self, batch_size_entity, batch_size_relation, seed=None):
        """
        """
        self.batch_size_entity = batch_size_entity
        self.batch_size_relation = batch_size_relation
        self.rng = np.random.RandomState(seed) # pylint: disable=no-member

    @property
    def supervised(self):
        """
        Distillation method will include the ground truth if the property supervised is set to True.
        """
        return True

    def get(self, mapping_entities, mapping_relations, positive_sample_size, **kwargs):
        """
        """
        entity_distribution_teacher = self.rng.choice(
            a=list(mapping_entities.keys()), size=self.batch_size_entity, replace=False)

        relation_distribution_teacher = self.rng.choice(
            a=list(mapping_relations.keys()),  size=self.batch_size_relation, replace=False)

        entity_distribution_student = [
            mapping_entities[entity] for entity in entity_distribution_teacher]

        relation_distribution_student = [
            mapping_relations[relation] for relation in relation_distribution_teacher]

        entity_distribution_teacher = torch.tensor(entity_distribution_teacher).view( # pylint: disable=not-callable
            1, self.batch_size_entity)

        entity_distribution_student = torch.tensor(entity_distribution_student).view( # pylint: disable=not-callable
            1, self.batch_size_entity)

        relation_distribution_teacher = torch.tensor(relation_distribution_teacher).view( # pylint: disable=not-callable
            1, self.batch_size_relation)

        relation_distribution_student = torch.tensor(relation_distribution_student).view( # pylint: disable=not-callable
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


class TopKSampling:
    """Top k sampling."""

    def __init__(self, teacher_entities, teacher_relations, student_entities,
        student_relations, teacher, batch_size_entity, batch_size_relation):
        self.batch_size_entity   = batch_size_entity
        self.batch_size_relation = batch_size_relation

        self.mapping_entities = collections.OrderedDict({
            i: student_entities[e] for e, i in teacher_entities.items()
            if e in student_entities})

        self.mapping_relations = collections.OrderedDict({
            i: student_relations[e] for e, i in teacher_relations.items()
            if e in student_relations})

        self.mapping_tree_entities_teacher = collections.defaultdict(int)
        self.mapping_tree_entities_student = collections.defaultdict(int)
        for i, (key, value) in enumerate(self.mapping_entities.items()):
            self.mapping_tree_entities_teacher[i] = key
            self.mapping_tree_entities_student[i] = value

        self.mapping_tree_relations_teacher = collections.defaultdict(int)
        self.mapping_tree_relations_student = collections.defaultdict(int)
        for i, (key, value) in enumerate(self.mapping_relations.items()):
            self.mapping_tree_relations_teacher[i] = key
            self.mapping_tree_relations_student[i] = value

        self.trees = {
            'entities' : faiss.IndexFlatL2(teacher.entity_dim),
            'relations': faiss.IndexFlatL2(teacher.relation_dim),
        }

        self.trees['entities'].add(
            teacher.entity_embedding.cpu().data.numpy()[list(self.mapping_entities.keys())])

        self.trees['relations'].add(
            teacher.relation_embedding.cpu().data.numpy()[list(self.mapping_relations.keys())])

    @property
    def supervised(self):
        return False

    def query_entities(self, x):
        _, neighbours = self.trees['entities'].search(x, k = self.batch_size_entity)
        return neighbours

    def query_relations(self, x):
        _, neighbours = self.trees['relations'].search(x, k = self.batch_size_relation)
        return neighbours

    def get(self, positive_sample, teacher, **kwargs):
        with torch.no_grad():
            score_head, score_relation, score_tail = teacher._top_k(positive_sample)

        score_head = score_head.cpu().data.numpy()
        score_relation = score_relation.cpu().data.numpy()
        score_tail = score_tail.cpu().data.numpy()

        score_head = score_head.reshape(
            positive_sample.shape[0], teacher.entity_dim)

        score_relation = score_relation.reshape(
            positive_sample.shape[0], teacher.relation_dim)

        score_tail = score_tail.reshape(
            positive_sample.shape[0], teacher.entity_dim)

        top_k_head = self.query_entities(x = score_head).flatten()
        top_k_relation = self.query_relations(x = score_relation).flatten()
        top_k_tail = self.query_entities(x = score_tail).flatten()

        head_distribution_teacher = torch.tensor(np.array( # pylint: disable=not-callable
            [self.mapping_tree_entities_teacher[x] for x in top_k_head]
        ).reshape(positive_sample.shape[0], self.batch_size_entity))

        relation_distribution_teacher = torch.tensor(np.array( # pylint: disable=not-callable
            [self.mapping_tree_relations_teacher[x] for x in top_k_relation]
        ).reshape(positive_sample.shape[0], self.batch_size_relation))

        tail_distribution_teacher = torch.tensor(np.array( # pylint: disable=not-callable
            [self.mapping_tree_entities_teacher[x] for x in top_k_tail]
        ).reshape(positive_sample.shape[0], self.batch_size_entity))

        head_distribution_student = torch.tensor(np.array( # pylint: disable=not-callable
            [self.mapping_tree_entities_student[x] for x in top_k_head]
        ).reshape(positive_sample.shape[0], self.batch_size_entity))

        relation_distribution_student = torch.tensor(np.array( # pylint: disable=not-callable
            [self.mapping_tree_relations_student[x] for x in top_k_relation]
        ).reshape(positive_sample.shape[0], self.batch_size_relation))

        tail_distribution_student = torch.tensor(np.array( # pylint: disable=not-callable
            [self.mapping_tree_entities_student[x] for x in top_k_tail]
        ).reshape(positive_sample.shape[0], self.batch_size_entity))

        return (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
            head_distribution_student, relation_distribution_student, tail_distribution_student)
