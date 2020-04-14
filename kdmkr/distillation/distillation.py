import torch

import collections
import itertools


__all__ = [
    'Distillation'
]


class Distillation:
    """
    Process sample of inputs datasets to distill knowledge from 1 to n knowledges bases.

    Example:

            :

            >>> from kdmkr import distillation

            >>> import torch

            >>> _ = torch.manual_seed(42)

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
            ...     student_relations=relations_student)

            # Which distillation method is available for a given triplet (h=1, r=1, t=1) from the
            # teacher knowledge resource?

            >>> print(distillation.distillation_mode(head=1, relation=1, tail=1))
            {'head': True, 'relation': True, 'tail': True}

            # Entity 0 of teacher is not in the KG of the student. We are only able to distill
            # on the head.
            >>> print(distillation.distillation_mode(head=0, relation=1, tail=1))
            {'head': True, 'relation': False, 'tail': False}

            >>> print(distillation.mini_batch_teacher_head(relation=1, tail=1))
            (tensor([[[1, 1, 1],
                      [2, 1, 1],
                      [3, 1, 1]]]), tensor([[[0, 1, 1]]]))

            >>> print(distillation.mini_batch_teacher_relation(head=1, tail=1))
            (tensor([[[1, 1, 1],
                      [1, 2, 1],
                      [1, 3, 1]]]), tensor([[[1, 0, 1]]]))

            >>> print(distillation.mini_batch_teacher_tail(head=1, relation=1))
            (tensor([[[1, 1, 1],
                      [1, 1, 2],
                      [1, 1, 3]]]), tensor([[[1, 1, 0]]]))

            >>> print(distillation.mini_batch_student_head(teacher_relation=1, teacher_tail=1))
            (tensor([[[0, 0, 0],
                      [1, 0, 0],
                      [2, 0, 0]]]), tensor([[[3, 0, 0]]]))

            >>> print(distillation.mini_batch_student_relation(teacher_head=1, teacher_tail=1))
            (tensor([[[0, 0, 0],
                      [0, 1, 0],
                      [0, 2, 0]]]), tensor([[[0, 3, 0]]]))

            >>> print(distillation.mini_batch_student_tail(teacher_head=1, teacher_relation=1))
            (tensor([[[0, 0, 0],
                      [0, 0, 1],
                      [0, 0, 2]]]), tensor([[[0, 0, 3]]]))

    """
    def __init__(self, teacher_entities, student_entities, teacher_relations, student_relations,
        ):
        self.teacher_entities = teacher_entities
        self.student_entities = student_entities
        self.teacher_relations = teacher_relations
        self.student_relations = student_relations

        # Common entities and relations pre-processing:
        self.mapping_entities = collections.OrderedDict({
            i: self.student_entities[e] for e, i in self.teacher_entities.items()
            if e in self.student_entities
        })

        self.mapping_relations = collections.OrderedDict({
            i: self.student_relations[e] for e, i in self.teacher_relations.items()
            if e in self.student_relations
        })

        self.common_head_batch_teacher = self.pre_compute_matrix(
            batch=[torch.tensor([e, 0, 0]) for e, _ in self.mapping_entities.items()], # pylint: disable=not-callable
            dim=(1, len(self.mapping_entities), 3)
        )

        self.common_tail_batch_teacher = self.pre_compute_matrix(
            batch=[torch.tensor([0, 0, e]) for e, _ in self.mapping_entities.items()], # pylint: disable=not-callable
            dim=(1, len(self.mapping_entities), 3)
        )

        self.common_relation_batch_teacher = self.pre_compute_matrix(
            batch=[torch.tensor([0, r, 0]) for r, _ in self.mapping_relations.items()], # pylint: disable=not-callable
            dim=(1, len(self.mapping_relations), 3)
        )

        self.common_head_batch_student = self.pre_compute_matrix(
            batch=[torch.tensor([e, 0, 0]) for _, e in self.mapping_entities.items()], # pylint: disable=not-callable
            dim=(1, len(self.mapping_entities), 3)
        )

        self.common_tail_batch_student = self.pre_compute_matrix(
            batch=[torch.tensor([0, 0, e]) for _, e in self.mapping_entities.items()], # pylint: disable=not-callable
            dim=(1, len(self.mapping_entities), 3)
        )

        self.common_relation_batch_student = self.pre_compute_matrix(
            batch=[torch.tensor([0, r, 0]) for _, r in self.mapping_relations.items()], # pylint: disable=not-callable
            dim=(1, len(self.mapping_relations), 3)
        )

        # Distinct entities and relations pre-processing:

        # Entities belonging to the student and teacher that are not shared across KG.
        self.distinct_teacher_entities = [i for e, i in self.teacher_entities.items()
            if e not in self.student_entities]

        self.distinct_student_entities = [i for e, i in self.student_entities.items()
            if e not in self.teacher_entities]

        # Relations belonging to the student and teacher that are not shared across KG.
        self.distinct_teacher_relations = [i for r, i in self.teacher_relations.items()
            if r not in self.student_relations]

        self.distinct_student_relations = [i for r, i in self.student_relations.items()
            if r not in self.teacher_relations]

        # Teacher:
        self.distinct_head_batch_teacher = self.pre_compute_matrix(
            batch=[torch.tensor([e, 0, 0]) for e in self.distinct_teacher_entities], # pylint: disable=not-callable
            dim=(1, len(self.distinct_teacher_entities), 3)
        )

        self.distinct_relation_batch_teacher = self.pre_compute_matrix(
            batch=[torch.tensor([0, r, 0]) for r in self.distinct_teacher_relations], # pylint: disable=not-callable
            dim=(1, len(self.distinct_teacher_relations), 3)
        )

        self.distinct_tail_batch_teacher = self.pre_compute_matrix(
            batch=[torch.tensor([0, 0, e]) for e in self.distinct_teacher_entities], # pylint: disable=not-callable
            dim=(1, len(self.distinct_teacher_entities), 3)
        )

        # Student:
        self.distinct_head_batch_student = self.pre_compute_matrix(
            batch=[torch.tensor([e, 0, 0]) for e in self.distinct_student_entities], # pylint: disable=not-callable
            dim=(1, len(self.distinct_student_entities), 3)
        )

        self.distinct_relation_batch_student = self.pre_compute_matrix(
            batch=[torch.tensor([0, r, 0]) for r in self.distinct_student_relations], # pylint: disable=not-callable
            dim=(1, len(self.distinct_student_relations), 3)
        )

        self.distinct_tail_batch_student = self.pre_compute_matrix(
            batch=[torch.tensor([0, 0, e]) for e in self.distinct_student_entities], # pylint: disable=not-callable
            dim=(1, len(self.distinct_student_entities), 3)
        )

    def mini_batch_teacher_head(self, relation, tail):
        self.common_head_batch_teacher[:,:,1] = relation
        self.common_head_batch_teacher[:,:,2] = tail

        if isinstance(self.distinct_head_batch_teacher, torch.Tensor):
            self.distinct_head_batch_teacher[:,:,1] = relation
            self.distinct_head_batch_teacher[:,:,2] = tail

        return self.common_head_batch_teacher, self.distinct_head_batch_teacher

    def mini_batch_teacher_relation(self, head, tail):
        self.common_relation_batch_teacher[:,:,0] = head
        self.common_relation_batch_teacher[:,:,2] = tail

        if isinstance(self.distinct_relation_batch_teacher, torch.Tensor):
            self.distinct_relation_batch_teacher[:,:,0] = head
            self.distinct_relation_batch_teacher[:,:,2] = tail

        return self.common_relation_batch_teacher, self.distinct_relation_batch_teacher

    def mini_batch_teacher_tail(self, head, relation):
        self.common_tail_batch_teacher[:,:,0] = head
        self.common_tail_batch_teacher[:,:,1] = relation

        if isinstance(self.distinct_tail_batch_teacher, torch.Tensor):
            self.distinct_tail_batch_teacher[:,:,0] = head
            self.distinct_tail_batch_teacher[:,:,1] = relation

        return self.common_tail_batch_teacher, self.distinct_tail_batch_teacher

    def mini_batch_student_head(self, teacher_relation, teacher_tail):
        relation = self.mapping_relations[teacher_relation]
        tail = self.mapping_entities[teacher_tail]

        self.common_head_batch_student[:,:,1] = relation
        self.common_head_batch_student[:,:,2] = tail

        if isinstance(self.distinct_head_batch_student, torch.Tensor):
            self.distinct_head_batch_student[:,:,1] = relation
            self.distinct_head_batch_student[:,:,2] = tail

        return self.common_head_batch_student, self.distinct_head_batch_student

    def mini_batch_student_relation(self, teacher_head, teacher_tail):
        head = self.mapping_entities[teacher_head]
        tail = self.mapping_entities[teacher_tail]

        self.common_relation_batch_student[:,:,0] = head
        self.common_relation_batch_student[:,:,2] = tail

        if isinstance(self.distinct_relation_batch_student, torch.Tensor):
            self.distinct_relation_batch_student[:,:,0] = head
            self.distinct_relation_batch_student[:,:,2] = tail

        return self.common_relation_batch_student, self.distinct_relation_batch_student

    def mini_batch_student_tail(self, teacher_head, teacher_relation):
        head = self.mapping_entities[teacher_head]
        relation = self.mapping_relations[teacher_relation]

        self.common_tail_batch_student[:,:,0] = head
        self.common_tail_batch_student[:,:,1] = relation

        if isinstance(self.distinct_tail_batch_student, torch.Tensor):
            self.distinct_tail_batch_student[:,:,0] = head
            self.distinct_tail_batch_student[:,:,1] = relation

        return self.common_tail_batch_student, self.distinct_tail_batch_student

    def pre_compute_matrix(self, batch, dim):
        if dim[1] > 0:
            batch = list(itertools.chain.from_iterable(batch))
            batch = torch.stack(batch).reshape(dim)
        else:
            batch = None
        return batch

    def distillation_mode(self, head, relation, tail):
        """
        Define which type of distillation is available.
        """
        head_available, relation_available, tail_available = False, False, False

        if head in self.mapping_entities:
            head_available = True

        if tail in self.mapping_entities:
            tail_available = True

        if relation in self.mapping_relations:
            relation_available = True

        return {
            'head': relation_available and tail_available,
            'relation': head_available and tail_available,
            'tail': head_available and relation_available,
        }
