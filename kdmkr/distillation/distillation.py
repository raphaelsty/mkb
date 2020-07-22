import torch

import collections
import copy
import itertools

import numpy as np

from .. import losses


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

            >>> uniform_sampling = distillation.UniformSampling(
            ... batch_size_entity   = 3,
            ... batch_size_relation = 3,
            ... seed = 42,
            ... )

            >>> distill = distillation.Distillation(
            ...     teacher_entities  = entities_teacher,
            ...     teacher_relations = relations_teacher,
            ...     student_entities  = entities_student,
            ...     student_relations = relations_student,
            ...     sampling          = uniform_sampling,
            ... )

            >>> print(distill.available(head=1, relation=1, tail=1))
            {'head': True, 'relation': True, 'tail': True}

            >>> print(distill.available(head=0, relation=1, tail=1))
            {'head': False, 'relation': False, 'tail': False}

            >>> (head_distribution_teacher, relation_distribution_teacher,
            ...  tail_distribution_teacher, head_distribution_student,
            ...  relation_distribution_student, tail_distribution_student) = uniform_sampling.get(
            ...    mapping_entities     = distill.mapping_entities,
            ...    mapping_relations    = distill.mapping_relations,
            ...    positive_sample_size = 1,
            ... )

            >>> print(head_distribution_teacher)
            tensor([[1., 2., 3.]])

            >>> print(relation_distribution_teacher)
            tensor([[2., 3., 1.]])

            >>> print(tail_distribution_teacher)
            tensor([[1., 2., 3.]])

            >>> print(head_distribution_student)
            tensor([[0., 1., 2.]])

            >>> print(relation_distribution_student)
            tensor([[1., 2., 0.]])

            >>> print(tail_distribution_student)
            tensor([[0., 1., 2.]])

            # Training sample from teacher KG:
            >>> head = 1
            >>> relation = 1
            >>> tail = 2

            >>> distillation_available = distill.available(
            ...    head=head, relation=relation, tail=tail)

            >>> print(distillation_available)
            {'head': True, 'relation': True, 'tail': True}

            >>> if distillation_available['head']:
            ...    teacher_head_tensor, student_head_tensor = distill.get_distillation_sample_head(
            ...         head_distribution_teacher=head_distribution_teacher,
            ...         head_distribution_student=head_distribution_student,
            ...         head_teacher=head, relation_teacher=relation, tail_teacher=tail)

            >>> if distillation_available['relation']:
            ...    teacher_relation_tensor, student_relation_tensor = distill.get_distillation_sample_relation(
            ...         relation_distribution_teacher=relation_distribution_teacher,
            ...         relation_distribution_student=relation_distribution_student,
            ...         head_teacher=head, relation_teacher=relation, tail_teacher=tail)

            >>> if distillation_available['tail']:
            ...    teacher_tail_tensor, student_tail_tensor = distill.get_distillation_sample_tail(
            ...         tail_distribution_teacher=tail_distribution_teacher,
            ...         tail_distribution_student=tail_distribution_student,
            ...         head_teacher=head, relation_teacher=relation, tail_teacher=tail)

            Test batch to distill head:
            >>> x = torch.Tensor(
            ...     [[[1., 1., 2.],
            ...       [2., 1., 2.],
            ...       [1., 1., 2.]]])

            # tensor([[1, 2, 3]])

            >>> torch.eq(teacher_head_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            >>> x = torch.Tensor(
            ...     [[[0., 0., 1.],
            ...       [1., 0., 1.],
            ...       [0., 0., 1.]]])

            >>> torch.eq(student_head_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            Test batch to distill relation:
            >>> x = torch.Tensor(
            ...     [[[1., 2., 2.],
            ...       [1., 3., 2.],
            ...       [1., 1., 2.]]])

            >>> torch.eq(teacher_relation_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            >>> x = torch.Tensor(
            ...     [[[0., 1., 1.],
            ...       [0., 2., 1.],
            ...       [0., 0., 1.]]])

            >>> torch.eq(student_relation_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            Test batch to distill tail:
            >>> x = torch.Tensor(
            ...     [[[1., 1., 1.],
            ...       [1., 1., 2.],
            ...       [1., 1., 2.]]])

            >>> torch.eq(teacher_tail_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

            >>> x = torch.Tensor(
            ...     [[[0., 0., 0.],
            ...       [0., 0., 1.],
            ...       [0., 0., 1.]]])

            >>> torch.eq(student_tail_tensor, x)
            tensor([[[True, True, True],
                     [True, True, True],
                     [True, True, True]]])

    """

    def __init__(self, teacher_entities, student_entities, teacher_relations, student_relations,
                 sampling, device='cpu'):
        self.teacher_entities = teacher_entities
        self.student_entities = student_entities
        self.teacher_relations = teacher_relations
        self.student_relations = student_relations
        self.sampling = sampling
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

    def available(self, head, relation, tail):
        """
        Define which type of distillation is available.
        """
        head_available, relation_available, tail_available = False, False, False
        distillation_available = {'head': False,
                                  'relation': False, 'tail': False}

        if head in self.mapping_entities:
            head_available = True

        if relation in self.mapping_relations:
            relation_available = True

        if tail in self.mapping_entities:
            tail_available = True

        # If the sampling is supervised: head, relation and tail must be shared by the student
        # and the teacher:
        if self.sampling.supervised:

            if head_available and relation_available and tail_available:

                distillation_available['head'] = True
                distillation_available['relation'] = True
                distillation_available['tail'] = True

        # If the sampling is unsupervised, we allow distillation on part of the triplet:
        else:

            if head_available and relation_available:
                distillation_available['tail'] = True

            if relation_available and tail_available:
                distillation_available['head'] = True

            if head_available and tail_available:
                distillation_available['relation'] = True

        return distillation_available

    def init_tensor(self, head, relation, tail, batch_size):
        x = torch.zeros((1, batch_size, 3))
        x[:, :, 0] = head
        x[:, :, 1] = relation
        x[:, :, 2] = tail
        return x

    def get_distillation_sample_head(self, head_distribution_teacher, head_distribution_student,
                                     head_teacher, relation_teacher, tail_teacher):
        # Teacher
        entity_distribution_teacher = copy.deepcopy(head_distribution_teacher)

        # Uniform sampling always include the ground truth:
        if self.sampling.supervised:
            entity_distribution_teacher[0][-1] = head_teacher

        tensor_head_teacher = self.init_tensor(
            head=entity_distribution_teacher,
            relation=relation_teacher,
            tail=tail_teacher,
            batch_size=self.sampling.batch_size_entity
        )

        relation_student = self.mapping_relations[relation_teacher]
        tail_student = self.mapping_entities[tail_teacher]

        entity_distribution_student = copy.deepcopy(head_distribution_student)

        # Uniform sampling always include the ground truth:
        if self.sampling.supervised:
            head_student = self.mapping_entities[head_teacher]
            entity_distribution_student[0][-1] = head_student

        tensor_head_student = self.init_tensor(
            head=entity_distribution_student,
            relation=relation_student,
            tail=tail_student,
            batch_size=self.sampling.batch_size_entity
        )

        return tensor_head_teacher, tensor_head_student

    def get_distillation_sample_relation(self, relation_distribution_teacher,
                                         relation_distribution_student, head_teacher, relation_teacher, tail_teacher):
        batch_size = relation_distribution_teacher.shape[0]

        # Teacher
        relation_distribution_teacher_copy = copy.deepcopy(
            relation_distribution_teacher)

        # Supervised samplers always include the ground truth:
        if self.sampling.supervised:
            relation_distribution_teacher_copy[0][-1] = relation_teacher

        tensor_relation_teacher = self.init_tensor(
            head=head_teacher,
            relation=relation_distribution_teacher_copy,
            tail=tail_teacher,
            batch_size=self.sampling.batch_size_relation
        )

        # Student
        head_student = self.mapping_entities[head_teacher]
        tail_student = self.mapping_entities[tail_teacher]

        relation_distribution_student_copy = copy.deepcopy(
            relation_distribution_student)

        # Supervised samplers always include the ground truth:
        if self.sampling.supervised:
            relation_student = self.mapping_relations[relation_teacher]
            relation_distribution_student_copy[0][-1] = relation_student

        tensor_relation_student = self.init_tensor(
            head=head_student,
            relation=relation_distribution_student_copy,
            tail=tail_student,
            batch_size=self.sampling.batch_size_relation
        )

        return tensor_relation_teacher, tensor_relation_student

    def get_distillation_sample_tail(self, tail_distribution_teacher, tail_distribution_student,
                                     head_teacher, relation_teacher, tail_teacher):
        # Teacher
        entity_distribution_teacher = copy.deepcopy(tail_distribution_teacher)

        # Supervised samplers always include the ground truth:
        if self.sampling.supervised:
            entity_distribution_teacher[0][-1] = tail_teacher

        tensor_tail_teacher = self.init_tensor(
            head=head_teacher,
            relation=relation_teacher,
            tail=entity_distribution_teacher,
            batch_size=self.sampling.batch_size_entity
        )

        # Student
        head_student = self.mapping_entities[head_teacher]
        relation_student = self.mapping_relations[relation_teacher]

        entity_distribution_student = copy.deepcopy(tail_distribution_student)

        # Uniform sampling always include the ground truth:
        if self.sampling.supervised:
            tail_student = self.mapping_entities[tail_teacher]
            entity_distribution_student[0][-1] = tail_student

        tensor_tail_student = self.init_tensor(
            head=head_student,
            relation=relation_student,
            tail=entity_distribution_student,
            batch_size=self.sampling.batch_size_entity
        )

        return tensor_tail_teacher, tensor_tail_student

    @classmethod
    def _stack_sample(cls, batch, batch_size, device):
        return torch.stack(batch).reshape(len(batch), batch_size, 3).to(device=device, dtype=int)

    def stack_entity(self, batch, device):
        """Convert a list of sample to 3 dimensionnal tensor"""
        return self._stack_sample(batch=batch, batch_size=self.sampling.batch_size_entity,
                                  device=device)

    def stack_relations(self, batch, device):
        """Convert a list of sample to 3 dimensionnal tensor"""
        return self._stack_sample(batch=batch, batch_size=self.sampling.batch_size_relation, device=device)

    def distill(self, teacher, student, positive_sample):
        """Apply distillation between a teacher and a student from a positive sample.
        Distill is a high level method. Power users can use lower level distillation methods.

            teacher (model): Teacher that will distill his knowledge.
            student (model): Student that will learn from the teacher.
            positive_sample (torch.Tensor): Positive example from the KB.

            Returns:

                If distillation available:
                    Distillation loss, Available set to True.
                Else:
                    None, Available set to False.

        Example:

                >>> from kdmkr import datasets
                >>> from kdmkr import distillation
                >>> from kdmkr import models

                >>> import torch
                >>> _ = torch.manual_seed(42)

                >>> import warnings
                >>> warnings.filterwarnings("ignore")

                >>> wn18rr = datasets.Wn18rr(batch_size=3, shuffle=True, seed=42)

                >>> teacher = models.RotatE(
                ...    hidden_dim = 3,
                ...    n_entity   = wn18rr.n_entity,
                ...    n_relation = wn18rr.n_relation,
                ...    gamma      = 6
                ... )

                >>> student = models.RotatE(
                ...    hidden_dim = 3,
                ...    n_entity   = wn18rr.n_entity,
                ...    n_relation = wn18rr.n_relation,
                ...    gamma      = 6
                ... )

                >>> distillation_process = distillation.Distillation(
                ...     teacher_entities  = wn18rr.entities,
                ...     student_entities  = wn18rr.entities,
                ...     teacher_relations = wn18rr.relations,
                ...     student_relations = wn18rr.relations,
                ...     sampling          = distillation.UniformSampling(
                ...         batch_size_entity   = 3,
                ...         batch_size_relation = 3,
                ...         seed                = 42,
                ...     ),
                ... )

                >>> positive_sample, weight, mode = next(wn18rr)

                >>> loss_distillation = distillation_process.distill(
                ...     teacher = teacher,
                ...     student = student,
                ...     positive_sample = positive_sample,
                ... )

                >>> loss_distillation
                {'head': tensor(0.2896, grad_fn=<MeanBackward0>), 'relation': tensor(0.0981, grad_fn=<MeanBackward0>), 'tail': tensor(0.4686, grad_fn=<MeanBackward0>)}

                >>> loss_student = (
                ...    loss_distillation['head'] +
                ...    loss_distillation['relation'] +
                ...    loss_distillation['tail']
                ... )

                >>> loss_student.backward()

                # Tok K sampling:
                >>> teacher = models.TransE(
                ...    hidden_dim = 3,
                ...    n_entity   = wn18rr.n_entity,
                ...    n_relation = wn18rr.n_relation,
                ...    gamma      = 6
                ... )

                >>> student = models.TransE(
                ...    hidden_dim = 3,
                ...    n_entity   = wn18rr.n_entity,
                ...    n_relation = wn18rr.n_relation,
                ...    gamma      = 6
                ... )

                >>> distillation_process = distillation.Distillation(
                ...     teacher_entities  = wn18rr.entities,
                ...     student_entities  = wn18rr.entities,
                ...     teacher_relations = wn18rr.relations,
                ...     student_relations = wn18rr.relations,
                ...     sampling = distillation.TopKSampling(
                ...         teacher_entities    = wn18rr.entities,
                ...         student_entities    = wn18rr.entities,
                ...         teacher_relations   = wn18rr.relations,
                ...         student_relations   = wn18rr.relations,
                ...         teacher             = teacher,
                ...         batch_size_entity   = 3,
                ...         batch_size_relation = 3,
                ...         n_random_entities   = 10,
                ...         n_random_relations  = 10,
                ...         seed                = 42,
                ...     ),
                ... )

                >>> positive_sample, weight, mode = next(wn18rr)

                >>> loss_distillation = distillation_process.distill(
                ...     teacher = teacher,
                ...     student = student,
                ...     positive_sample = positive_sample,
                ... )

                >>> loss_distillation
                {'head': tensor(0.2777, grad_fn=<MeanBackward0>), 'relation': tensor(0.1471, grad_fn=<MeanBackward0>), 'tail': tensor(0.1874, grad_fn=<MeanBackward0>)}

                >>> loss_student = (
                ...    loss_distillation['head'] +
                ...    loss_distillation['relation'] +
                ...    loss_distillation['tail']
                ... )

                >>> loss_student.backward()

        """
        batch_head_teacher = []
        batch_head_student = []

        batch_relation_teacher = []
        batch_relation_student = []

        batch_tail_teacher = []
        batch_tail_student = []

        loss_distillation = {'head': None, 'relation': None, 'tail': None}

        (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
            head_distribution_student, relation_distribution_student, tail_distribution_student
         ) = self.sampling.get(**{
             'positive_sample': positive_sample,
             'mapping_entities': self.mapping_entities,
             'mapping_relations': self.mapping_relations,
             'positive_sample_size': positive_sample.shape[0],
             'teacher': teacher,
         })

        for i, (head, relation, tail) in enumerate(positive_sample):

            head, relation, tail = head.item(), relation.item(), tail.item()

            distillation_available = self.available(
                head=head, relation=relation, tail=tail)

            if distillation_available['head']:

                tensor_head_teacher, tensor_head_student = self.get_distillation_sample_head(
                    head_distribution_teacher=head_distribution_teacher[i].unsqueeze(
                        dim=0),
                    head_distribution_student=head_distribution_student[i].unsqueeze(
                        dim=0),
                    head_teacher=head, relation_teacher=relation, tail_teacher=tail
                )

                batch_head_teacher.append(tensor_head_teacher)
                batch_head_student.append(tensor_head_student)

            if distillation_available['relation']:

                tensor_relation_teacher, tensor_relation_student = self.get_distillation_sample_relation(
                    relation_distribution_teacher=relation_distribution_teacher[i].unsqueeze(
                        dim=0),
                    relation_distribution_student=relation_distribution_student[i].unsqueeze(
                        dim=0),
                    head_teacher=head, relation_teacher=relation, tail_teacher=tail
                )

                batch_relation_teacher.append(tensor_relation_teacher)
                batch_relation_student.append(tensor_relation_student)

            if distillation_available['tail']:

                tensor_tail_teacher, tensor_tail_student = self.get_distillation_sample_tail(
                    tail_distribution_teacher=tail_distribution_teacher[i].unsqueeze(
                        dim=0),
                    tail_distribution_student=tail_distribution_student[i].unsqueeze(
                        dim=0),
                    head_teacher=head, relation_teacher=relation, tail_teacher=tail
                )

                batch_tail_teacher.append(tensor_tail_teacher)
                batch_tail_student.append(tensor_tail_student)

        # Compute loss dedicated to relations if any relations or tails from the input sample are
        # shared in student and teacher kgs.
        if batch_head_teacher:
            teacher_head_tensor = self.stack_entity(
                batch_head_teacher, device=self.device)
            student_head_tensor = self.stack_entity(
                batch_head_student, device=self.device)
            # Disable gradients for teacher:
            with torch.no_grad():
                scores_head_teacher = teacher.distill(teacher_head_tensor)

            # Distillation loss of heads
            loss_distillation['head'] = losses.KlDivergence()(
                teacher_score=scores_head_teacher,
                student_score=student.distill(student_head_tensor)
            )

        # Compute loss dedicated to relations if any heads or tails from the input sample are shared
        # in student and teacher kgs.
        if batch_relation_teacher:
            teacher_relation_tensor = self.stack_relations(
                batch_relation_teacher, device=self.device)
            student_relation_tensor = self.stack_relations(
                batch_relation_student, device=self.device)
            # Disable gradients for teacher:
            with torch.no_grad():
                scores_relation_teacher = teacher.distill(
                    teacher_relation_tensor)

            # Distillation loss of relations.
            loss_distillation['relation'] = losses.KlDivergence()(
                teacher_score=scores_relation_teacher,
                student_score=student.distill(student_relation_tensor)
            )

        # Compute loss dedicated to tails if any heads or relations from the input sample are shared
        #  in student and teachers kgs.
        if batch_tail_teacher:
            teacher_tail_tensor = self.stack_entity(
                batch_tail_teacher, device=self.device)
            student_tail_tensor = self.stack_entity(
                batch_tail_student, device=self.device)
            # Disable gradients for teacher:
            with torch.no_grad():
                scores_tail_teacher = teacher.distill(teacher_tail_tensor)

            # Distillation loss of tails.
            loss_distillation['tail'] = losses.KlDivergence()(
                teacher_score=scores_tail_teacher,
                student_score=student.distill(student_tail_tensor)
            )

        return loss_distillation
