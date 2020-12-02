import torch
import tqdm
import numpy as np

import collections

from ..models import TransE

__all__ = ['FastTopKSampling', 'TopKSampling', 'TopKSamplingTransE']


class FastTopKSampling:
    """Top k sampling which pre-compute scores.

    Example:

        >>> from mkb import distillation
        >>> from mkb import datasets
        >>> from mkb import models
        >>> from mkb import utils
        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset_teacher = datasets.CountriesS1(batch_size = 2, seed = 42, shuffle=False)
        >>> dataset_student = datasets.CountriesS2(batch_size = 2, seed = 42, shuffle=False)

        >>> teacher = models.RotatE(
        ...     entities = dataset_teacher.entities,
        ...     relations = dataset_teacher.relations,
        ...     gamma = 3,
        ...     hidden_dim = 4
        ... )

        >>> distillation = distillation.FastTopKSampling(
        ...     teacher_relations = dataset_teacher.relations,
        ...     teacher_entities = dataset_teacher.entities,
        ...     student_entities = dataset_student.entities,
        ...     student_relations = dataset_student.relations,
        ...     batch_size_entity = 4,
        ...     batch_size_relation = 1,
        ...     n_random_entities = 1,
        ...     n_random_relations = 0,
        ...     seed = 42,
        ...     teacher = teacher,
        ...     dataset_teacher = dataset_teacher,
        ... )

        >>> sample = next(iter(dataset_teacher))['sample']

        >>> sample
        tensor([[  0,   0, 266],
                [  1,   1,  56]])

        >>> (
        ...    head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
        ...    head_distribution_student, relation_distribution_student, tail_distribution_student
        ... ) = distillation.get(sample = sample, teacher = teacher)

        >>> head_distribution_teacher
        tensor([[197,  50,  75, 176,  30],
                [ 10, 240, 251,   3,  30]])

        >>> relation_distribution_teacher
        tensor([[0],
                [1]])

        >>> tail_distribution_teacher
        tensor([[269, 210, 270, 261,  30],
                [120, 160, 212, 244,  30]])

        >>> head_distribution_student
        tensor([[186,  47,  70, 166,  28],
                [ 10, 229, 240,   3,  28]])

        >>> relation_distribution_student
        tensor([[0],
                [1]])

        >>> tail_distribution_student
        tensor([[269, 198, 270, 256,  28],
                [111, 149, 201, 234,  28]])

        Check if the top k computes top k and not bottom k.

        Check top k on heads:
        >>> heads = torch.tensor([
        ...    [197, 0, 266],
        ...    [50,  0, 266],
        ...    [75,  0, 266],
        ...    [176, 0, 266],
        ...    [30,  0, 266],
        ... ])

        >>> teacher(heads)
        tensor([[ 1.0877],
                [ 0.6982],
                [ 0.6696],
                [ 0.4555],
                [-0.8312]], grad_fn=<ViewBackward>)

        >>> for e, _ in distillation.mapping_entities.items():
        ...     score = teacher(torch.tensor([[e, 0, 266]]))
        ...     if score > 0.4555:
        ...         print(e, score)
        50 tensor([[0.6982]], grad_fn=<ViewBackward>)
        75 tensor([[0.6696]], grad_fn=<ViewBackward>)
        197 tensor([[1.0877]], grad_fn=<ViewBackward>)

        Check top k on relations:
        >>> relations = torch.tensor([
        ...    [
        ...         [0,  0, 266],
        ...         [0,  1, 266],
        ...     ],
        ...     [
        ...         [1,  0, 56],
        ...         [1,  1, 56],
        ...      ]
        ... ])

        >>> teacher(relations)
        tensor([[-2.5582, -3.1826],
                [-3.6232, -2.4322]], grad_fn=<ViewBackward>)

        Relation 0 is the right top 1 for the triple (0, ?, 266).
        Relation 1 is the right top 1 for the triple (1, ?, 56).

        Check top k on tails:
        >>> tails = torch.tensor([
        ...    [0, 0, 269],
        ...    [0, 0, 210],
        ...    [0, 0, 270],
        ...    [0, 0, 261],
        ...    [0, 0, 30],
        ... ])

        >>> teacher(tails)
        tensor([[ 1.5890],
                [ 0.3337],
                [ 0.1993],
                [ 0.1206],
                [-3.0354]], grad_fn=<ViewBackward>)

        >>> for e, _ in distillation.mapping_entities.items():
        ...     score = teacher(torch.tensor([[0, 0, e]]))
        ...     if score > 0.1206:
        ...         print(e, score)
        210 tensor([[0.3337]], grad_fn=<ViewBackward>)
        269 tensor([[1.5890]], grad_fn=<ViewBackward>)
        270 tensor([[0.1993]], grad_fn=<ViewBackward>)

    """

    def __init__(self, teacher_entities, teacher_relations, student_entities, student_relations,
                 batch_size_entity, batch_size_relation, n_random_entities, n_random_relations,
                 dataset_teacher, teacher, device='cpu', seed=None, **kwargs):

        if isinstance(teacher, TransE):
            base_method = TopKSamplingTransE
        else:
            base_method = TopKSampling

        distillator = base_method(**{
            'teacher_entities': teacher_entities,
            'teacher_relations': teacher_relations,
            'student_entities': student_entities,
            'student_relations': student_relations,
            'batch_size_entity': batch_size_entity,
            'batch_size_relation': batch_size_relation,
            'n_random_entities': 0,
            'n_random_relations': 0,
            'device': device,
            'seed': seed,
            'teacher': teacher,
        })

        self.mapping_entities = distillator.mapping_entities
        self.mapping_relations = distillator.mapping_relations

        self.batch_size_entity_top_k = batch_size_entity
        self.batch_size_relation_top_k = batch_size_relation

        self.n_random_entities = n_random_entities
        self.n_random_relations = n_random_relations

        self.dict_head_distribution_teacher = {}
        self.dict_relation_distribution_teacher = {}
        self.dict_tail_distribution_teacher = {}

        self.dict_head_distribution_student = {}
        self.dict_relation_distribution_student = {}
        self.dict_tail_distribution_student = {}

        self._rng = np.random.RandomState(seed)

        for data in tqdm.tqdm(dataset_teacher, position=0):

            if data['mode'] == 'head-batch':

                (head_distribution_teacher, relation_distribution_teacher,
                    tail_distribution_teacher, head_distribution_student,
                    relation_distribution_student, tail_distribution_student
                 ) = distillator.get(data['sample'], teacher)

                for i, sample in enumerate(data['sample']):

                    h, r, t = sample[0].item(
                    ), sample[1].item(), sample[2].item()

                    self.dict_head_distribution_teacher[f'{r}_{t}'] = head_distribution_teacher[i]
                    self.dict_relation_distribution_teacher[f'{h}_{t}'] = relation_distribution_teacher[i]
                    self.dict_tail_distribution_teacher[f'{h}_{r}'] = tail_distribution_teacher[i]

                    self.dict_head_distribution_student[f'{r}_{t}'] = head_distribution_student[i]
                    self.dict_relation_distribution_student[f'{h}_{t}'] = relation_distribution_student[i]
                    self.dict_tail_distribution_student[f'{h}_{r}'] = tail_distribution_student[i]

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

    def get(self, sample, **kwargs):
        (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
            head_distribution_student, relation_distribution_student, tail_distribution_student
         ) = [], [], [], [], [], []

        distribution = []

        for x in sample:
            h, r, t = x[0].item(), x[1].item(), x[2].item()

            head_distribution_teacher.append(
                self.dict_head_distribution_teacher[f'{r}_{t}'])
            relation_distribution_teacher.append(
                self.dict_relation_distribution_teacher[f'{h}_{t}'])
            tail_distribution_teacher.append(
                self.dict_tail_distribution_teacher[f'{h}_{r}'])

            head_distribution_student.append(
                self.dict_head_distribution_student[f'{r}_{t}'])
            relation_distribution_student.append(
                self.dict_relation_distribution_student[f'{h}_{t}'])
            tail_distribution_student.append(
                self.dict_tail_distribution_student[f'{h}_{r}'])

        head_distribution_teacher = torch.stack(
            head_distribution_teacher, dim=0)
        relation_distribution_teacher = torch.stack(
            relation_distribution_teacher, dim=0)
        tail_distribution_teacher = torch.stack(
            tail_distribution_teacher, dim=0)

        head_distribution_student = torch.stack(
            head_distribution_student, dim=0)
        relation_distribution_student = torch.stack(
            relation_distribution_student, dim=0)
        tail_distribution_student = torch.stack(
            tail_distribution_student, dim=0)

        (
            head_distribution_teacher, relation_distribution_teacher,
            tail_distribution_teacher, head_distribution_student,
            relation_distribution_student, tail_distribution_student
        ) = _randomize_distribution(
            sample=sample,
            n_random_entities=self.n_random_entities,
            n_random_relations=self.n_random_relations,
            mapping_entities=self.mapping_entities,
            mapping_relations=self.mapping_relations,
            _rng=self._rng,
            head_distribution_teacher=head_distribution_teacher,
            relation_distribution_teacher=relation_distribution_teacher,
            tail_distribution_teacher=tail_distribution_teacher,
            head_distribution_student=head_distribution_student,
            relation_distribution_student=relation_distribution_student,
            tail_distribution_student=tail_distribution_student,
        )

        return (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
                head_distribution_student, relation_distribution_student, tail_distribution_student)


class TopKSampling:
    """TopKSampling is dedicated to distill top entities and relations of a given sample.

    Creates 3 tensors for the student and the teacher for each single training sample. Those tensors
    are made of indexes and allows to computes distribution probability on a subset of entities and
    relations of the knowledge graph. Top k sampling returns the most probable entities and
    relations from the teacher point of view for incoming training samples.

    It is recommended to add randomly selected entities and relationships in addition to the top k.
    This allows entities and relationships that are not linked in the embedding space to be removed.

    Parameters:
        teacher_entities (dict): Entities of the teacher with labels as keys and index as values.
        student_entities (dict): Entities of the student with labels as keys and index as values.
        teacher_relations (dict): Relations of the student with labels as keys and index as values.
        student_relations (dict): Relations of the student with labels as keys and index as values.
        batch_size_entity (int): Number of entities to consider to compute distribution probability
            when using distillation.
        batch_size_relation (int): Number of relations to consider to compute distribution
            probability when using distillation.
        n_random_entities (int): Number of random entities to add in the distribution probability.
        n_random_relations (int): Number of random relations to add in the distribution probability.
        'device
        seed (int): Random state.


    Example:

        >>> from mkb import distillation
        >>> from mkb import datasets
        >>> from mkb import models
        >>> from mkb import utils
        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset_teacher = datasets.CountriesS1(batch_size = 2, seed = 42, shuffle=False)
        >>> dataset_student = datasets.CountriesS2(batch_size = 2, seed = 42, shuffle=False)

        >>> teacher = models.RotatE(
        ...     entities = dataset_teacher.entities,
        ...     relations = dataset_teacher.relations,
        ...     gamma = 3,
        ...     hidden_dim = 4
        ... )

        >>> distillation = distillation.TopKSampling(
        ...     teacher_relations = dataset_teacher.relations,
        ...     teacher_entities = dataset_teacher.entities,
        ...     student_entities = dataset_student.entities,
        ...     student_relations = dataset_student.relations,
        ...     batch_size_entity = 4,
        ...     batch_size_relation = 1,
        ...     n_random_entities = 1,
        ...     n_random_relations = 0,
        ...     seed = 42,
        ... )

        >>> sample = next(iter(dataset_teacher))['sample']

        >>> sample
        tensor([[  0,   0, 266],
                [  1,   1,  56]])

        >>> (
        ...    head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
        ...    head_distribution_student, relation_distribution_student, tail_distribution_student
        ... ) = distillation.get(sample = sample, teacher = teacher)

        >>> head_distribution_teacher
        tensor([[197,  50,  75, 176,  30],
                [ 10, 240, 251,   3,  30]])

        >>> relation_distribution_teacher
        tensor([[0],
                [1]])

        >>> tail_distribution_teacher
        tensor([[269, 210, 270, 261,  30],
                [120, 160, 212, 244,  30]])

        >>> head_distribution_student
        tensor([[186,  47,  70, 166,  28],
                [ 10, 229, 240,   3,  28]])

        >>> relation_distribution_student
        tensor([[0],
                [1]])

        >>> tail_distribution_student
        tensor([[269, 198, 270, 256,  28],
                [111, 149, 201, 234,  28]])

        Check if the top k computes top k and not bottom k.

        Check top k on heads:
        >>> heads = torch.tensor([
        ...    [197, 0, 266],
        ...    [50,  0, 266],
        ...    [75,  0, 266],
        ...    [176, 0, 266],
        ...    [30,  0, 266],
        ... ])

        >>> teacher(heads)
        tensor([[ 1.0877],
                [ 0.6982],
                [ 0.6696],
                [ 0.4555],
                [-0.8312]], grad_fn=<ViewBackward>)

        >>> for e, _ in distillation.mapping_entities.items():
        ...     score = teacher(torch.tensor([[e, 0, 266]]))
        ...     if score > 0.4555:
        ...         print(e, score)
        50 tensor([[0.6982]], grad_fn=<ViewBackward>)
        75 tensor([[0.6696]], grad_fn=<ViewBackward>)
        197 tensor([[1.0877]], grad_fn=<ViewBackward>)

        Check top k on relations:
        >>> relations = torch.tensor([
        ...    [
        ...         [0,  0, 266],
        ...         [0,  1, 266],
        ...     ],
        ...     [
        ...         [1,  0, 56],
        ...         [1,  1, 56],
        ...      ]
        ... ])

        >>> teacher(relations)
        tensor([[-2.5582, -3.1826],
                [-3.6232, -2.4322]], grad_fn=<ViewBackward>)

        Relation 0 is the right top 1 for the triple (0, ?, 266).
        Relation 1 is the right top 1 for the triple (1, ?, 56).

        Check top k on tails:
        >>> tails = torch.tensor([
        ...    [0, 0, 269],
        ...    [0, 0, 210],
        ...    [0, 0, 270],
        ...    [0, 0, 261],
        ...    [0, 0, 30],
        ... ])

        >>> teacher(tails)
        tensor([[ 1.5890],
                [ 0.3337],
                [ 0.1993],
                [ 0.1206],
                [-3.0354]], grad_fn=<ViewBackward>)

        >>> for e, _ in distillation.mapping_entities.items():
        ...     score = teacher(torch.tensor([[0, 0, e]]))
        ...     if score > 0.1206:
        ...         print(e, score)
        210 tensor([[0.3337]], grad_fn=<ViewBackward>)
        269 tensor([[1.5890]], grad_fn=<ViewBackward>)
        270 tensor([[0.1993]], grad_fn=<ViewBackward>)


    """

    def __init__(
            self, teacher_entities, teacher_relations, student_entities, student_relations,
            batch_size_entity, batch_size_relation, n_random_entities, n_random_relations, device='cpu',
            seed=None, **kwargs
    ):

        self.batch_size_entity_top_k = batch_size_entity
        self.batch_size_relation_top_k = batch_size_relation

        self.n_random_entities = n_random_entities
        self.n_random_relations = n_random_relations

        self.device = device
        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

        self.mapping_entities = collections.OrderedDict({
            i: student_entities[e] for e, i in teacher_entities.items()
            if e in student_entities
        })

        self.mapping_relations = collections.OrderedDict({
            i: student_relations[r] for r, i in teacher_relations.items()
            if r in student_relations
        })

        self.entities_teacher_prediction = torch.tensor([
            [e for e, _ in self.mapping_entities.items()]
        ], dtype=int)

        self.entities_teacher_selection = torch.tensor(
            [e for e, _ in self.mapping_entities.items()],
            dtype=int
        )

        self.entities_student = torch.tensor(
            [e for _, e in self.mapping_entities.items()], dtype=int
        )

        self.relations_teacher = torch.tensor(
            [r for r, _ in self.mapping_relations.items()], dtype=int
        )

        self.relations_student = torch.tensor(
            [r for _, r in self.mapping_relations.items()], dtype=int
        )

        self.tensor_relations_teacher = torch.stack(
            [
                torch.zeros(len(self.mapping_relations), dtype=int),
                self.relations_teacher,
                torch.zeros(len(self.mapping_relations), dtype=int),
            ],
            dim=1
        )

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

    def get(self, sample, teacher, **kwargs):

        teacher = teacher.eval()

        head_distribution_teacher = []
        head_distribution_student = []
        relation_distribution_teacher = []
        relation_distribution_student = []
        tail_distribution_teacher = []
        tail_distribution_student = []

        for head, relation, tail in sample:

            head, relation, tail = head.item(), relation.item(), tail.item()

            self.tensor_relations_teacher[:, 0] = head
            self.tensor_relations_teacher[:, 2] = tail

            rank_heads = self._get_rank_entities(
                teacher=teacher,
                sample=torch.tensor([[head, relation, tail]]),
                mode='head-batch',
                entities=self.entities_teacher_prediction,
                batch_size=self.batch_size_entity_top_k,
                device=self.device
            )

            rank_relations = self._get_rank_relations(
                teacher=teacher,
                sample=self.tensor_relations_teacher,
                batch_size=self.batch_size_relation_top_k,
                device=self.device
            )

            rank_tails = self._get_rank_entities(
                teacher=teacher,
                sample=torch.tensor([[head, relation, tail]]),
                mode='tail-batch',
                entities=self.entities_teacher_prediction,
                batch_size=self.batch_size_entity_top_k,
                device=self.device
            )

            head_distribution_teacher.append(
                self.entities_teacher_selection[rank_heads]
            )

            head_distribution_student.append(
                self.entities_student[rank_heads]
            )

            relation_distribution_teacher.append(
                self.relations_student[rank_relations]
            )

            relation_distribution_student.append(
                self.relations_student[rank_relations]
            )

            tail_distribution_teacher.append(
                self.entities_teacher_selection[rank_tails]
            )

            tail_distribution_student.append(
                self.entities_student[rank_tails]
            )

        teacher = teacher.train()

        head_distribution_teacher = torch.stack(
            head_distribution_teacher,
            dim=0
        )

        relation_distribution_teacher = torch.stack(
            relation_distribution_teacher,
            dim=0
        )

        tail_distribution_teacher = torch.stack(
            tail_distribution_teacher,
            dim=0
        )

        head_distribution_student = torch.stack(
            head_distribution_student,
            dim=0
        )

        relation_distribution_student = torch.stack(
            relation_distribution_student,
            dim=0
        )

        tail_distribution_student = torch.stack(
            tail_distribution_student,
            dim=0
        )

        (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
         head_distribution_student, relation_distribution_student, tail_distribution_student
         ) = _randomize_distribution(
            sample=sample,
            n_random_entities=self.n_random_entities,
            n_random_relations=self.n_random_relations,
            mapping_entities=self.mapping_entities,
            mapping_relations=self.mapping_relations,
            _rng=self._rng,
            head_distribution_teacher=head_distribution_teacher,
            relation_distribution_teacher=relation_distribution_teacher,
            tail_distribution_teacher=tail_distribution_teacher,
            head_distribution_student=head_distribution_student,
            relation_distribution_student=relation_distribution_student,
            tail_distribution_student=tail_distribution_student,
        )

        return (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
                head_distribution_student, relation_distribution_student, tail_distribution_student)

    @classmethod
    def _get_rank_relations(cls, teacher, sample, batch_size, device):
        with torch.no_grad():
            return torch.argsort(
                teacher(sample.to(device)),
                descending=True,
                dim=0
            ).flatten()[:batch_size]

    @classmethod
    def _get_rank_entities(cls, teacher, sample, entities, mode, batch_size, device):
        """Speed up computation of best candidates entities using negative sample mechanism."""
        with torch.no_grad():
            return torch.argsort(
                teacher(
                    sample.to(device),
                    entities.to(device),
                    mode
                ),
                dim=1,
                descending=True
            )[:, 0:batch_size].flatten()


class TopKSamplingTransE:
    """Unsupervised top k sampling dedicated to distillation.

    Creates 3 tensors for the student and the teacher for each single training sample. Those tensors
    are made of indexes and allows to computes distribution probability on a subset of entities and
    relations of the knowledge graph. Top k sampling returns the most probable entities and
    relations from the teacher point of view for incoming training samples.

    It is recommended to add randomly selected entities and relationships in addition to the top k.
    This allows entities and relationships that are not linked in the embedding space to be removed.

    Parameters:
        teacher_entities (dict): Entities of the teacher with labels as keys and index as values.
        student_entities (dict): Entities of the student with labels as keys and index as values.
        teacher_relations (dict): Relations of the student with labels as keys and index as values.
        student_relations (dict): Relations of the student with labels as keys and index as values.
        teacher (mkb.models): The model who plays the role of the teacher.
        batch_size_entity (int): Number of entities to consider to compute distribution probability
            when using distillation.
        batch_size_relation (int): Number of relations to consider to compute distribution
            probability when using distillation.
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
            seed=None, **kwargs):
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

    def get(self, sample, teacher, **kwargs):
        with torch.no_grad():
            score_head, score_relation, score_tail = teacher._top_k(
                sample)

        score_head = score_head.cpu().data.numpy()
        score_relation = score_relation.cpu().data.numpy()
        score_tail = score_tail.cpu().data.numpy()

        score_head = score_head.reshape(
            sample.shape[0], teacher.entity_dim)

        score_relation = score_relation.reshape(
            sample.shape[0], teacher.relation_dim)

        score_tail = score_tail.reshape(
            sample.shape[0], teacher.entity_dim)

        top_k_head = self.query_entities(x=score_head).flatten()
        top_k_relation = self.query_relations(x=score_relation).flatten()
        top_k_tail = self.query_entities(x=score_tail).flatten()

        head_distribution_teacher = torch.LongTensor(np.array(
            [self.mapping_tree_entities_teacher[x] for x in top_k_head]
        ).reshape(sample.shape[0], self.batch_size_entity_top_k))

        relation_distribution_teacher = torch.LongTensor(np.array(
            [self.mapping_tree_relations_teacher[x] for x in top_k_relation]
        ).reshape(sample.shape[0], self.batch_size_relation_top_k))

        tail_distribution_teacher = torch.LongTensor(np.array(
            [self.mapping_tree_entities_teacher[x] for x in top_k_tail]
        ).reshape(sample.shape[0], self.batch_size_entity_top_k))

        head_distribution_student = torch.LongTensor(np.array(
            [self.mapping_tree_entities_student[x] for x in top_k_head]
        ).reshape(sample.shape[0], self.batch_size_entity_top_k))

        relation_distribution_student = torch.LongTensor(np.array(
            [self.mapping_tree_relations_student[x] for x in top_k_relation]
        ).reshape(sample.shape[0], self.batch_size_relation_top_k))

        tail_distribution_student = torch.LongTensor(np.array(
            [self.mapping_tree_entities_student[x] for x in top_k_tail]
        ).reshape(sample.shape[0], self.batch_size_entity_top_k))

        (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
         head_distribution_student, relation_distribution_student, tail_distribution_student
         ) = _randomize_distribution(
            sample=sample,
            n_random_entities=self.n_random_entities,
            n_random_relations=self.n_random_relations,
            mapping_entities=self.mapping_entities,
            mapping_relations=self.mapping_relations,
            _rng=self._rng,
            head_distribution_teacher=head_distribution_teacher,
            relation_distribution_teacher=relation_distribution_teacher,
            tail_distribution_teacher=tail_distribution_teacher,
            head_distribution_student=head_distribution_student,
            relation_distribution_student=relation_distribution_student,
            tail_distribution_student=tail_distribution_student,
        )

        return (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
                head_distribution_student, relation_distribution_student, tail_distribution_student)


def _randomize_distribution(
    sample, n_random_entities, n_random_relations, mapping_entities, mapping_relations,
    _rng, head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
    head_distribution_student, relation_distribution_student, tail_distribution_student
):
    """Randomize distribution in ouput of top k. Append n random entities and n random relations
    to distillation distribution from teacher and sudent shared entities and relations.
    """
    if n_random_entities > 0:

        random_entities_teacher = _rng.choice(
            list(mapping_entities.keys()),
            size=n_random_entities,
            replace=False
        )

        random_entities_student = torch.LongTensor(
            [[mapping_entities[i] for i in random_entities_teacher]]
        )

        random_entities_teacher = torch.cat(
            sample.shape[0] * [torch.LongTensor([random_entities_teacher])])

        random_entities_student = torch.cat(
            sample.shape[0] * [random_entities_student])

        head_distribution_teacher = torch.cat(
            [head_distribution_teacher, random_entities_teacher], dim=1)

        head_distribution_student = torch.cat(
            [head_distribution_student, random_entities_student], dim=1)

        tail_distribution_teacher = torch.cat(
            [tail_distribution_teacher, random_entities_teacher], dim=1)

        tail_distribution_student = torch.cat(
            [tail_distribution_student, random_entities_student], dim=1)

    if n_random_relations > 0:

        random_relations_teacher = _rng.choice(
            list(mapping_relations.keys()),
            size=n_random_relations,
            replace=False
        )

        random_relations_student = torch.LongTensor([[
            mapping_relations[i] for i in random_relations_teacher]])

        random_relations_teacher = torch.cat(
            sample.shape[0] * [torch.LongTensor([random_relations_teacher])])

        random_relations_student = torch.cat(
            sample.shape[0] * [random_relations_student])

        relation_distribution_teacher = torch.cat(
            [relation_distribution_teacher, random_relations_teacher], dim=1)

        relation_distribution_student = torch.cat(
            [relation_distribution_student, random_relations_student], dim=1)

    return (head_distribution_teacher, relation_distribution_teacher, tail_distribution_teacher,
            head_distribution_student, relation_distribution_student, tail_distribution_student)
