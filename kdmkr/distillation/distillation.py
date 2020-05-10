import torch

import collections
import copy
import itertools

import numpy as np

from .. import loss


__all__ = [
    'Distillation'
]


class Distillation:
    """
    Process sample of inputs datasets to distill knowledge from 1 to n knowledges bases with
    uniform subsampling.

    Example:

            >>> from kdmkr import distillation

            >>> entities_teacher = {
            ...    'e0': 0,
            ...    'e1': 1,
            ...    'e2': 2,
            ...    'e3': 3,
            ... }

            >>> relations_teacher = {
            ...     'r0': 0,
            ...     'r1': 1,
            ...     'r2': 2,
            ...     'r3': 3,
            ... }

            >>> entities_student = {
            ...    'e1': 0,
            ...    'e2': 1,
            ...    'e3': 2,
            ...    'e4': 3,
            ... }

            >>> relations_student = {
            ...     'r1': 0,
            ...     'r2': 1,
            ...     'r3': 2,
            ...     'r5': 3,
            ... }

            >>> train_teacher = [
            ...     (2, 0, 3),
            ...     (2, 1, 3),
            ... ]

            >>> train_student = [
            ...     (3, 1, 4),
            ...     (3, 2, 4),
            ... ]

            >>> distillation = distillation.Distillation(teacher_entities=entities_teacher,
            ...     student_entities=entities_student, teacher_relations=relations_teacher,
            ...     student_relations=relations_student, batch_size_entity=3, batch_size_relation=3,
            ...     seed=42)

            >>> print(distillation.available(head=1, relation=1, tail=1))
            True

            >>> print(distillation.available(head=0, relation=1, tail=1))
            False

            >>> print(distillation.uniform_subsampling())
            (tensor([[3, 1, 3]]), tensor([[3, 1, 1]]), tensor([[2, 0, 2]]), tensor([[2, 0, 0]]))

            # Training sample from teacher KG:
            >>> head     = 1
            >>> relation = 1
            >>> tail     = 2

            >>> (
            ...    entity_distribution_teacher,
            ...    relation_distribution_teacher,
            ...    entity_distribution_student,
            ...    relation_distribution_student
            ... ) = distillation.uniform_subsampling()

            >>> distillation_available = distillation.available(
            ...    head=head, relation=relation, tail=tail)

            >>> print(distillation_available)
            True

            >>> if distillation_available:
            ...    teacher_head_tensor, student_head_tensor = distillation.get_distillation_sample_head(
            ...         entity_distribution_teacher=entity_distribution_teacher,
            ...         entity_distribution_student=entity_distribution_student,
            ...         head_teacher=head, relation_teacher=relation, tail_teacher=tail)
            ...
            ...    teacher_relation_tensor, student_relation_tensor = distillation.get_distillation_sample_relation(
            ...         relation_distribution_teacher=relation_distribution_teacher,
            ...         relation_distribution_student=relation_distribution_student,
            ...         head_teacher=head, relation_teacher=relation, tail_teacher=tail)
            ...
            ...    teacher_tail_tensor, student_tail_tensor = distillation.get_distillation_sample_tail(
            ...         entity_distribution_teacher=entity_distribution_teacher,
            ...         entity_distribution_student=entity_distribution_student,
            ...         head_teacher=head, relation_teacher=relation, tail_teacher=tail)

            Test batch to distill head:
            >>> x = torch.tensor(
            ...     [[[1., 1., 2.],
            ...       [2., 1., 2.],
            ...       [3., 1., 2.]]])

            >>> torch.eq(teacher_head_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            >>> x = torch.tensor(
            ...     [[[0., 0., 1.],
            ...       [1., 0., 1.],
            ...       [2., 0., 1.]]])

            >>> torch.eq(student_head_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            Test batch to distill relation:
            >>> x = torch.tensor(
            ...     [[[1., 1., 2.],
            ...       [1., 3., 2.],
            ...       [1., 3., 2.]]])

            >>> torch.eq(teacher_relation_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            >>> x = torch.tensor(
            ...     [[[0., 0., 1.],
            ...       [0., 2., 1.],
            ...       [0., 2., 1.]]])

            >>> torch.eq(student_relation_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            Test batch to distill tail:
            >>> x = torch.tensor(
            ...     [[[1., 1., 2.],
            ...       [1., 1., 2.],
            ...       [1., 1., 3.]]])

            >>> torch.eq(teacher_tail_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            >>> x = torch.tensor(
            ...     [[[0., 0., 1.],
            ...       [0., 0., 1.],
            ...       [0., 0., 2.]]])

            >>> torch.eq(student_tail_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

    """
    def __init__(self, teacher_entities, student_entities, teacher_relations, student_relations,
        batch_size_entity, batch_size_relation, seed=None, device='cpu'):
        self.teacher_entities = teacher_entities
        self.student_entities = student_entities
        self.teacher_relations = teacher_relations
        self.student_relations = student_relations
        self.batch_size_entity = batch_size_entity
        self.batch_size_relation = batch_size_relation
        self.seed = seed
        self.device = device

        # Common entities and relations pre-processing:
        self.mapping_entities = collections.OrderedDict({
            i: self.student_entities[e] for e, i in self.teacher_entities.items()
            if e in self.student_entities
        })

        self.mapping_relations = collections.OrderedDict({
            i: self.student_relations[e] for e, i in self.teacher_relations.items()
            if e in self.student_relations
        })

        if self.seed is not None:
            self._rng = np.random.seed(self.seed)

    def available(self, head, relation, tail):
        """
        Define which type of distillation is available.
        """
        head_available, relation_available, tail_available = False, False, False

        if (head in self.mapping_entities
            and tail in self.mapping_entities
            and relation in self.mapping_relations):
            return True
        else:
            return False

    def init_tensor(self, head, relation, tail, batch_size):
        x = torch.zeros((1, batch_size, 3))
        x[:,:,0] = head
        x[:,:,1] = relation
        x[:,:,2] = tail
        return x

    def uniform_subsampling(self):
        """Init minibatch dedicated to distillation with uniform subsampling for student and
        teacher.
        """
        entity_distribution_teacher = np.random.choice(
            a=list(self.mapping_entities.keys()), size=self.batch_size_entity,
        )

        relation_distribution_teacher = np.random.choice(
            a=list(self.mapping_relations.keys()),  size=self.batch_size_relation
        )

        entity_distribution_student = [
            self.mapping_entities[entity] for entity in entity_distribution_teacher]

        relation_distribution_student = [
            self.mapping_relations[relation] for relation in relation_distribution_teacher]

        entity_distribution_teacher = torch.tensor(entity_distribution_teacher).view( # pylint: disable=not-callable
            1, self.batch_size_entity)

        entity_distribution_student = torch.tensor(entity_distribution_student).view( # pylint: disable=not-callable
            1, self.batch_size_entity)

        relation_distribution_teacher = torch.tensor(relation_distribution_teacher).view( # pylint: disable=not-callable
            1, self.batch_size_relation)

        relation_distribution_student = torch.tensor(relation_distribution_student).view( # pylint: disable=not-callable
            1, self.batch_size_relation)

        return (entity_distribution_teacher, relation_distribution_teacher,
            entity_distribution_student, relation_distribution_student)

    def get_distillation_sample_head(self, entity_distribution_teacher, entity_distribution_student,
        head_teacher, relation_teacher, tail_teacher):
        # Teacher
        head_distribution_teacher       = copy.deepcopy(entity_distribution_teacher)
        head_distribution_teacher[0][0] = head_teacher

        tensor_head_teacher = self.init_tensor(
            head       = head_distribution_teacher,
            relation   = relation_teacher,
            tail       = tail_teacher,
            batch_size = self.batch_size_entity
        )

        # Student
        head_student     = self.mapping_entities[head_teacher]
        relation_student = self.mapping_relations[relation_teacher]
        tail_student     = self.mapping_entities[tail_teacher]

        head_distribution_student       = copy.deepcopy(entity_distribution_student)
        head_distribution_student[0][0] = head_student

        tensor_head_student = self.init_tensor(
            head       = head_distribution_student,
            relation   = relation_student,
            tail       = tail_student,
            batch_size = self.batch_size_entity
        )

        return tensor_head_teacher, tensor_head_student

    def get_distillation_sample_relation(self, relation_distribution_teacher,
        relation_distribution_student, head_teacher, relation_teacher, tail_teacher):
        batch_size = relation_distribution_teacher.shape[0]

        # Teacher
        relation_distribution_teacher_copy = copy.deepcopy(relation_distribution_teacher)
        relation_distribution_teacher_copy[0][0] = relation_teacher

        tensor_relation_teacher = self.init_tensor(
            head       = head_teacher,
            relation   = relation_distribution_teacher_copy,
            tail       = tail_teacher,
            batch_size = self.batch_size_relation
        )

        # Student
        head_student     = self.mapping_entities[head_teacher]
        relation_student = self.mapping_relations[relation_teacher]
        tail_student     = self.mapping_entities[tail_teacher]

        relation_distribution_student_copy = copy.deepcopy(relation_distribution_student)
        relation_distribution_student_copy[0][0] = relation_student

        tensor_relation_student = self.init_tensor(
            head       = head_student,
            relation   = relation_distribution_student_copy,
            tail       = tail_student,
            batch_size = self.batch_size_relation
        )

        return tensor_relation_teacher, tensor_relation_student

    def get_distillation_sample_tail(self, entity_distribution_teacher, entity_distribution_student,
        head_teacher, relation_teacher, tail_teacher):
        # Teacher
        tail_distribution_teacher       = copy.deepcopy(entity_distribution_teacher)
        tail_distribution_teacher[0][0] = tail_teacher

        tensor_tail_teacher = self.init_tensor(
            head       = head_teacher,
            relation   = relation_teacher,
            tail       = tail_distribution_teacher,
            batch_size = self.batch_size_entity
        )

        # Student
        head_student     = self.mapping_entities[head_teacher]
        relation_student = self.mapping_relations[relation_teacher]
        tail_student     = self.mapping_entities[tail_teacher]

        tail_distribution_student       = copy.deepcopy(entity_distribution_student)
        tail_distribution_student[0][0] = tail_student

        tensor_tail_student = self.init_tensor(
            head       = head_student,
            relation   = relation_student,
            tail       = tail_distribution_student,
            batch_size = self.batch_size_entity
        )

        return tensor_tail_teacher, tensor_tail_student

    @classmethod
    def _stack_sample(cls, batch, batch_size, device):
        return torch.stack(batch).reshape(len(batch), batch_size, 3).to(device=device, dtype=int)

    def stack_entity(self, batch, device):
        """Convert a list of sample to 3 dimensionnal tensor"""
        return self._stack_sample(batch=batch, batch_size=self.batch_size_entity, device=device)

    def stack_relations(self, batch, device):
        """Convert a list of sample to 3 dimensionnal tensor"""
        return self._stack_sample(batch=batch, batch_size=self.batch_size_relation, device=device)

    def distill(self, teacher, student, positive_sample):
        """Apply distillation between a teacher and a student from a positive sample."""
        batch_head_teacher = []
        batch_head_student = []

        batch_relation_teacher = []
        batch_relation_student = []

        batch_tail_teacher = []
        batch_tail_student = []

        (entity_distribution_teacher, relation_distribution_teacher,
            entity_distribution_student, relation_distribution_student
    ) = self.uniform_subsampling()

        for head, relation, tail in positive_sample:

            head, relation, tail = head.item(), relation.item(), tail.item()

            distillation_available = self.available(head=head, relation=relation, tail=tail)

            if distillation_available:

                tensor_head_teacher, tensor_head_student = self.get_distillation_sample_head(
                    entity_distribution_teacher=entity_distribution_teacher,
                    entity_distribution_student=entity_distribution_student,
                    head_teacher=head, relation_teacher=relation, tail_teacher=tail
                )

                tensor_relation_teacher, tensor_relation_student = self.get_distillation_sample_relation(
                    relation_distribution_teacher=relation_distribution_teacher,
                    relation_distribution_student=relation_distribution_student,
                    head_teacher=head, relation_teacher=relation, tail_teacher=tail
                )

                tensor_tail_teacher, tensor_tail_student = self.get_distillation_sample_tail(
                    entity_distribution_teacher=entity_distribution_teacher,
                    entity_distribution_student=entity_distribution_student,
                    head_teacher=head, relation_teacher=relation, tail_teacher=tail
                )

                batch_head_teacher.append(tensor_head_teacher)
                batch_head_student.append(tensor_head_student)

                batch_relation_teacher.append(tensor_relation_teacher)
                batch_relation_student.append(tensor_relation_student)

                batch_tail_teacher.append(tensor_tail_teacher)
                batch_tail_student.append(tensor_tail_student)

        if batch_head_teacher or batch_relation_teacher or batch_tail_teacher:

            teacher_head_tensor = self.stack_entity(batch_head_teacher, device=self.device)
            student_head_tensor = self.stack_entity(batch_head_student, device=self.device)

            teacher_relation_tensor = self.stack_relations(batch_relation_teacher, device=self.device)
            student_relation_tensor = self.stack_relations(batch_relation_student, device=self.device)

            teacher_tail_tensor = self.stack_entity(batch_tail_teacher, device=self.device)
            student_tail_tensor = self.stack_entity(batch_tail_student, device=self.device)

            # Distillation loss of heads
            loss_head = loss.KlDivergence()(
                teacher_score=teacher.distill(teacher_head_tensor),
                student_score=student.distill(student_head_tensor)
            )

            # Distillation loss of relations.
            loss_relation = loss.KlDivergence()(
                teacher_score=teacher.distill(teacher_relation_tensor),
                student_score=student.distill(student_relation_tensor)
            )

            # Distillation loss of tails.
            loss_tail = loss.KlDivergence()(
                teacher_score=teacher.distill(teacher_tail_tensor),
                student_score=student.distill(student_tail_tensor)
            )

            # The loss of the student is equal to the sum of all losses.
            loss_distillation = loss_head + loss_relation + loss_tail

            return loss_distillation, True
        else:
            return None, False
