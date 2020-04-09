import torch

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
            >>> from kdmkr import stream

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
            ... }

            >>> train_teacher = [
            ...     (2, 0, 3),
            ...     (2, 1, 3),
            ... ]

            >>> train_student = [
            ...     (3, 1, 4),
            ...     (3, 2, 4),
            ... ]

            >>> dataset_teacher = stream.FetchDataset(train=train_teacher, entities=entities_teacher,
            ...    relations=relations_teacher, negative_sample_size=1, batch_size=1, seed=42)

            >>> dataset_student = stream.FetchDataset(train=train_student,  entities=entities_student,
            ...    relations=relations_student, negative_sample_size=1, batch_size=1, seed=42)

            >>> distillation = distillation.Distillation(teacher_entities=entities_teacher,
            ...     student_entities=entities_student, teacher_relations=relations_teacher,
            ...     student_relations=relations_student)

            >>> print(distillation.mini_batch_teacher_head(relation=0, tail=1))
            tensor([[[0, 0, 1],
                     [1, 0, 1],
                     [2, 0, 1],
                     [3, 0, 1]]])

            >>> print(distillation.mini_batch_teacher_relation(head=0, tail=1))
            tensor([[[0, 0, 1],
                     [0, 1, 1],
                     [0, 2, 1],
                     [0, 3, 1]]])

            >>> print(distillation.mini_batch_teacher_tail(head=0, relation=1))
            tensor([[[0, 1, 0],
                     [0, 1, 1],
                     [0, 1, 2],
                     [0, 1, 3]]])

            >>> print(distillation.mini_batch_student_head(relation=0, tail=1))
            tensor([[[0, 0, 1],
                     [1, 0, 1],
                     [2, 0, 1],
                     [3, 0, 1]]])

            >>> print(distillation.mini_batch_student_relation(head=0, tail=1))
            tensor([[[0, 0, 1],
                     [0, 1, 1],
                     [0, 2, 1]]])

            >>> print(distillation.mini_batch_teacher_tail(head=0, relation=1))
            tensor([[[0, 1, 0],
                     [0, 1, 1],
                     [0, 1, 2],
                     [0, 1, 3]]])

    """
    def __init__(self, teacher_entities, student_entities, teacher_relations, student_relations,
        ):
        self.teacher_entities = teacher_entities
        self.student_entities = student_entities
        self.teacher_relations = teacher_relations
        self.student_relations = student_relations

        self.mapping_teacher_student = {
            i: self.student_entities[e] for e, i in self.teacher_entities.items()
            if e in self.student_entities
        }

        self.mapping_student_teacher = {r: e for e, r in self.mapping_teacher_student.items()}

        self.head_batch_teacher = self.pre_compute_matrix(
            batch = [
                torch.tensor([e, 0, 0]) for e in range(len(teacher_entities)) # pylint: disable=not-callable
            ],
            dim = (1, len(teacher_entities), 3)
        )

        self.relation_batch_teacher = self.pre_compute_matrix(
            batch = [
                torch.tensor([0, r, 0]) for r in range(len(teacher_relations)) # pylint: disable=not-callable
            ],
            dim = (1, len(teacher_relations), 3)
        )

        self.tail_batch_teacher = self.pre_compute_matrix(
            batch = [
                torch.tensor([0, 0, t]) for t in range(len(teacher_entities)) # pylint: disable=not-callable
            ],
            dim = (1, len(teacher_entities), 3)
        )

        self.head_batch_student = self.pre_compute_matrix(
            batch = [
                torch.tensor([e, 0, 0]) for e in range(len(student_entities)) # pylint: disable=not-callable
            ],
            dim = (1, len(student_entities), 3)
        )

        self.relation_batch_student = self.pre_compute_matrix(
            batch = [
                torch.tensor([0, r, 0]) for r in range(len(student_relations)) # pylint: disable=not-callable
            ],
            dim = (1, len(student_relations), 3)
        )

        self.tail_batch_student = self.pre_compute_matrix(
            batch = [
                torch.tensor([0, 0, t]) for t in range(len(student_entities)) # pylint: disable=not-callable
            ],
            dim = (1, len(student_entities), 3)
        )

    def mini_batch_teacher_head(self, relation, tail):
        self.head_batch_teacher[:,:,1] = relation
        self.head_batch_teacher[:,:,2] = tail
        return self.head_batch_teacher

    def mini_batch_teacher_relation(self, head, tail):
        self.relation_batch_teacher[:,:,0] = head
        self.relation_batch_teacher[:,:,2] = tail
        return self.relation_batch_teacher

    def mini_batch_teacher_tail(self, head, relation):
        self.tail_batch_teacher[:,:,0] = head
        self.tail_batch_teacher[:,:,1] = relation
        return self.tail_batch_teacher

    def mini_batch_student_head(self, relation, tail):
        self.head_batch_student[:,:,1] = relation
        self.head_batch_student[:,:,2] = tail
        return self.head_batch_student

    def mini_batch_student_relation(self, head, tail):
        self.relation_batch_student[:,:,0] = head
        self.relation_batch_student[:,:,2] = tail
        return self.relation_batch_student

    def mini_batch_student_tail(self, head, relation):
        self.tail_batch_student[:,:,0] = head
        self.tail_batch_student[:,:,1] = relation
        return self.tail_batch_student

    def pre_compute_matrix(self, batch, dim):
        batch = list(itertools.chain.from_iterable(batch))
        batch = torch.stack(batch).reshape(dim)
        return batch
