import torch

import numpy as np

import collections

__all__ = ['UniformSampling', 'TopKSampling']


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

    def __init__(self, batch_size_entity, batch_size_relation, seed=None):
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


class TopKSampling:
    """Unsupervised top k sampling dedicated to distillation.

    Creates 3 tensors for the student and the teacher for each single training sample. Those tensors
    are made of indexes and allows to computes distribution probability on a subset of entities and
    relations of the knowledge graph. Top k sampling returns the most probable entities and
    relations from the teacher point of view for incoming training samples.

    Parameters:
        batch_size_entity (int): Number of entities to consider to compute distribution probability
            when using distillation.
        batch_size_relation (int): Number of relations to consider to compute distribution
            probability when using distillation.
        teacher_entities (dict): Entities of the teacher with labels as keys and index as values.
        student_entities (dict): Entities of the student with labels as keys and index as values.
        teacher_relations (dict): Relations of the student with labels as keys and index as values.
        student_relations (dict): Relations of the student with labels as keys and index as values.
        n_random_entities (int): Number of random entities to add in the distribution probability.
        n_random_relations (int): Number of random relations to add in the distribution probability.
        seed (int): Random state.

    .. tip::
        Adding random entities and relations allows models to move away from entities/relationships
        in the embeddings space that aren't related to each other.

    """

    def __init__(
            self, teacher_entities, teacher_relations, student_entities, student_relations, teacher,
            batch_size_entity, batch_size_relation, n_random_entities, n_random_relations,
            seed=None):
        import faiss  # pylint: disable=import-error

        self.batch_size_entity_top_k = batch_size_entity
        self.batch_size_relation_top_k = batch_size_relation
        self.n_random_entities = n_random_entities
        self.n_random_relations = n_random_relations
        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

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
            'entities': faiss.IndexFlatL2(teacher.entity_dim),
            'relations': faiss.IndexFlatL2(teacher.relation_dim),
        }

        self.trees['entities'].add(
            teacher.entity_embedding.cpu().data.numpy()[list(self.mapping_entities.keys())])

        self.trees['relations'].add(
            teacher.relation_embedding.cpu().data.numpy()[list(self.mapping_relations.keys())])

    @property
    def supervised(self):
        """Do not include the ground truth."""
        return False

    @property
    def batch_size_entity(self):
        return self.batch_size_entity_top_k + self.n_random_entities

    @property
    def batch_size_relation(self):
        return self.batch_size_relation_top_k + self.n_random_relations

    def query_entities(self, x):
        _, neighbours = self.trees['entities'].search(
            x, k=self.batch_size_entity_top_k)
        return neighbours

    def query_relations(self, x):
        _, neighbours = self.trees['relations'].search(
            x, k=self.batch_size_relation_top_k)
        return neighbours

    def get(self, positive_sample, teacher, **kwargs):
        with torch.no_grad():
            score_head, score_relation, score_tail = teacher._top_k(
                positive_sample)

        score_head = score_head.cpu().data.numpy()
        score_relation = score_relation.cpu().data.numpy()
        score_tail = score_tail.cpu().data.numpy()

        score_head = score_head.reshape(
            positive_sample.shape[0], teacher.entity_dim)

        score_relation = score_relation.reshape(
            positive_sample.shape[0], teacher.relation_dim)

        score_tail = score_tail.reshape(
            positive_sample.shape[0], teacher.entity_dim)

        top_k_head = self.query_entities(x=score_head).flatten()
        top_k_relation = self.query_relations(x=score_relation).flatten()
        top_k_tail = self.query_entities(x=score_tail).flatten()

        head_distribution_teacher = torch.Tensor(np.array(
            [self.mapping_tree_entities_teacher[x] for x in top_k_head]
        ).reshape(positive_sample.shape[0], self.batch_size_entity_top_k))

        relation_distribution_teacher = torch.Tensor(np.array(
            [self.mapping_tree_relations_teacher[x] for x in top_k_relation]
        ).reshape(positive_sample.shape[0], self.batch_size_relation_top_k))

        tail_distribution_teacher = torch.Tensor(np.array(
            [self.mapping_tree_entities_teacher[x] for x in top_k_tail]
        ).reshape(positive_sample.shape[0], self.batch_size_entity_top_k))

        head_distribution_student = torch.Tensor(np.array(
            [self.mapping_tree_entities_student[x] for x in top_k_head]
        ).reshape(positive_sample.shape[0], self.batch_size_entity_top_k))

        relation_distribution_student = torch.Tensor(np.array(
            [self.mapping_tree_relations_student[x] for x in top_k_relation]
        ).reshape(positive_sample.shape[0], self.batch_size_relation_top_k))

        tail_distribution_student = torch.Tensor(np.array(
            [self.mapping_tree_entities_student[x] for x in top_k_tail]
        ).reshape(positive_sample.shape[0], self.batch_size_entity_top_k))

        (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
         head_distribution_student, relation_distribution_student, tail_distribution_student
         ) = self.randomize_distribution(
            positive_sample=positive_sample,
            head_distribution_teacher=head_distribution_teacher,
            relation_distribution_teacher=relation_distribution_teacher,
            tail_distribution_teacher=tail_distribution_teacher,
            head_distribution_student=head_distribution_student,
            relation_distribution_student=relation_distribution_student,
            tail_distribution_student=tail_distribution_student,
        )

        return (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
                head_distribution_student, relation_distribution_student, tail_distribution_student)

    def randomize_distribution(
        self, positive_sample, head_distribution_teacher, relation_distribution_teacher,
        tail_distribution_teacher, head_distribution_student, relation_distribution_student,
        tail_distribution_student
    ):
        """Randomize distribution in ouput of top k. Append n random entities and n random relations
        to distillation distribution from teacher and sudent shared entities and relations.
        """
        if self.n_random_entities > 0:

            random_entities_teacher = self._rng.choice(
                list(self.mapping_entities.keys()),
                size=self.n_random_entities,
                replace=False
            )

            random_entities_student = torch.Tensor(
                [[self.mapping_entities[i] for i in random_entities_teacher]])

            random_entities_teacher = torch.cat(
                positive_sample.shape[0] * [torch.Tensor([random_entities_teacher])])

            random_entities_student = torch.cat(
                positive_sample.shape[0] * [random_entities_student])

            head_distribution_teacher = torch.cat(
                [head_distribution_teacher, random_entities_teacher], dim=1)

            head_distribution_student = torch.cat(
                [head_distribution_student, random_entities_student], dim=1)

            tail_distribution_teacher = torch.cat(
                [tail_distribution_teacher, random_entities_teacher], dim=1)

            tail_distribution_student = torch.cat(
                [tail_distribution_student, random_entities_student], dim=1)

        if self.n_random_relations > 0:

            random_relations_teacher = self._rng.choice(
                list(self.mapping_relations.keys()),
                size=self.n_random_relations,
                replace=False
            )

            random_relations_student = torch.Tensor([[
                self.mapping_relations[i] for i in random_relations_teacher]])

            random_relations_teacher = torch.cat(
                positive_sample.shape[0] * [torch.Tensor([random_relations_teacher])])

            random_relations_student = torch.cat(
                positive_sample.shape[0] * [random_relations_student])

            relation_distribution_teacher = torch.cat(
                [relation_distribution_teacher, random_relations_teacher], dim=1)

            relation_distribution_student = torch.cat(
                [relation_distribution_student, random_relations_student], dim=1)

        return (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
                head_distribution_student, relation_distribution_student, tail_distribution_student)
