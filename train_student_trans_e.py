from kdmkr import distillation
from kdmkr import evaluation
from kdmkr import loss
from kdmkr import model
from kdmkr import stream
from kdmkr import utils
from kdmkr import sampling

from creme import stats

import numpy as np

import pickle
import torch
import tqdm
import sys
import yaml

device = 'cuda'

configuration_file = sys.argv[2]

with open(f'{configuration_file}') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    configuration = yaml.load(file, Loader=yaml.FullLoader)

train_file = configuration['train_file']
valid_file = configuration['valid_file']
test_file  = configuration['test_file']

entities_file  = configuration['entities_file']
relations_file = configuration['relations_file']

folder_training_set   = configuration['folder_training_set']
folder_evaluation_set = configuration['folder_evaluation_set']

hidden_dim = configuration['hidden_dim']
batch_size = configuration['batch_size']
gamma      = configuration['gamma']
max_step   = configuration['max_step']

batch_size_entity   = configuration['batch_size_entity']
batch_size_relation = configuration['batch_size_relation']
n_random_entities   = configuration['n_random_entities']
n_random_relations  = configuration['n_random_relations']

teacher_name  = configuration['teacher_name']

# Load teacher
with open(f'./models_id/{teacher_name}.pickle', 'rb') as handle:

    teacher = pickle.load(handle)
teacher = teacher.eval()
teacher = teacher.to(device)

student_name = f'TransE_{train_file}'

directory           = f'/users/iris/rsourty/experiments/kdmkr/kdmkr/datasets/{folder_training_set}'
train_file_path     = f'{directory}/{train_file}.csv'
valid_file_path     = f'{directory}/{valid_file}.csv'
test_file_path      = f'{directory}/{test_file}.csv'
entities_file_path  = f'{directory}/{entities_file}.json'
relations_file_path = f'{directory}/{relations_file}.json'

dataset = stream.FetchDataset(
    train = utils.read_csv(file_path=train_file_path),
    valid = utils.read_csv(file_path=valid_file_path),
    test  = utils.read_csv(file_path=test_file_path),
    batch_size = batch_size,
    shuffle    = True,
    seed       = 42,
    entities   = utils.read_json(entities_file_path),
    relations  = utils.read_json(relations_file_path),
)

train_file_path = f'/users/iris/rsourty/experiments/kdmkr/kdmkr/datasets/{folder_evaluation_set}/train.csv'

validation = evaluation.Evaluation(
    all_true_triples = (
        utils.read_csv(file_path=train_file_path) +
        utils.read_csv(file_path=valid_file_path) +
        utils.read_csv(file_path=test_file_path)
    ),
    entities   = utils.read_json(entities_file_path),
    relations  = utils.read_json(relations_file_path),
    batch_size = 2,
    device     = device,
)

student = model.TransE(
    hidden_dim = hidden_dim,
    n_entity   = dataset.n_entity,
    n_relation = dataset.n_relation,
    gamma      = gamma,
)

student.train()

student = student.to(device)

optimizer_student = torch.optim.Adam(
    filter(lambda p: p.requires_grad, student.parameters()), lr = 0.00005)

top_k = distillation.TopKSampling(
    teacher_entities    = utils.read_json(entities_file_path),
    teacher_relations   = utils.read_json(relations_file_path),
    student_entities    = utils.read_json(entities_file_path),
    student_relations   = utils.read_json(relations_file_path),
    teacher             = teacher,
    batch_size_entity   = batch_size_entity,
    batch_size_relation = batch_size_relation,
    n_random_entities   = n_random_entities,
    n_random_relations  = n_random_relations,
)

distillation_process = distillation.Distillation(
    teacher_entities  = utils.read_json(entities_file_path),
    student_entities  = utils.read_json(entities_file_path),
    teacher_relations = utils.read_json(relations_file_path),
    student_relations = utils.read_json(relations_file_path),
    sampling          = top_k,
    device            = device,
)

bar = tqdm.tqdm(range(1, max_step), position=0)

metric = stats.RollingMean(1000)

for step in bar:

    optimizer_student.zero_grad()

    positive_sample, weight, mode = next(dataset)

    positive_sample = positive_sample.to(device)

    loss_distillation = distillation_process.distill(
        positive_sample = positive_sample,
        teacher         = teacher,
        student         = student,
    )

    loss_student = (loss_distillation['head'] + loss_distillation['relation'] +
        loss_distillation['tail'])

    loss_student.backward()
    optimizer_student.step()
    metric.update(loss_student.item())

    if step % 5 == 0:
        bar.set_description(f'Adversarial loss: {metric.get():6f}')

    if step % 3000 == 0:

        student = student.eval()

        scores = validation.eval(model=student, dataset=utils.read_csv(file_path=test_file_path))

        student = student.train()

        print(scores)

        # Set path HERE
        with open(f'./models/student_{student_name}_{scores}.pickle', 'wb') as handle:

            pickle.dump(student, handle, protocol = pickle.HIGHEST_PROTOCOL)
