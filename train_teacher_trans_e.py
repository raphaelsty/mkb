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

entities_file         = configuration['entities_file']
relations_file        = configuration['relations_file']

folder_training_set   = configuration['folder_training_set']
folder_evaluation_set = configuration['folder_evaluation_set']

hidden_dim            = int(configuration['hidden_dim'])
batch_size            = int(configuration['batch_size'])
negative_sample_size  = int(configuration['negative_sample_size'])
gamma                 = float(configuration['gamma'])
alpha                 = float(configuration['alpha'])
max_step              = int(configuration['max_step'])

model_name = f'TransE_{train_file}'

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

negative_sampling = sampling.NegativeSampling(
    size          = negative_sample_size,
    train_triples = dataset.train,
    entities      = dataset.entities,
    relations     = dataset.relations,
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

model = model.TransE(
    hidden_dim = hidden_dim,
    n_entity   = dataset.n_entity,
    n_relation = dataset.n_relation,
    gamma      = gamma,
)

model = model.to(device)

optimizer_teacher = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr = 0.00005)

bar = tqdm.tqdm(range(1, max_step), position=0)

metric = stats.RollingMean(1000)

model.train()

for step in bar:

    optimizer_teacher.zero_grad()

    positive_sample, weight, mode = next(dataset)
    negative_sample = negative_sampling.generate(positive_sample, mode)
    positive_sample = positive_sample.to(device)
    negative_sample = negative_sample.to(device)

    weight = weight.to(device)

    positive_score = model(sample=positive_sample)
    negative_score = model(sample=(positive_sample, negative_sample), mode=mode)

    loss_teacher = loss.Adversarial()(positive_score, negative_score, weight, alpha=alpha)

    loss_teacher.backward()
    optimizer_teacher.step()
    metric.update(loss_teacher.item())

    if step % 5 == 0:
        bar.set_description(f'Adversarial loss: {metric.get():6f}')

    if step % 3000 == 0:

        model = model.eval()

        scores = validation.eval(model=model, dataset=utils.read_csv(file_path=test_file_path))

        model = model.train()

        print(scores)

        # Set path HERE
        with open(f'./models/{model_name}_{scores}.pickle', 'wb') as handle:

            pickle.dump(model, handle, protocol = pickle.HIGHEST_PROTOCOL)
