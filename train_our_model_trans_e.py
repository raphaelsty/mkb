from kdmkr import distillation
from kdmkr import evaluation
from kdmkr import loss
from kdmkr import model
from kdmkr import stream
from kdmkr import utils
from kdmkr import sampling
from creme import stats

import numpy as np

import collections
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

dataset_name           = configuration['dataset_name']             # wn18rr / fb15k237
hidden_dim             = configuration['hidden_dim']          # 500 / 1000
batch_size             = configuration['batch_size']          # 256 / 512
negative_sampling_size = configuration['negative_sampling_size']
max_step               = configuration['max_step']            # 40000
batch_size_entity      = configuration['batch_size_entity']   # 20
batch_size_relation    = configuration['batch_size_relation'] # 11 / 20
n_random_entities      = configuration['n_random_entities']   # 20
n_random_relations     = configuration['n_random_relations']  # 0 / 20
nb_teacher             = configuration['nb_teacher']          # 2 / 3
gamma                  = configuration['gamma']
evaluation_folder      = configuration['evaluation_folder']
alpha_adversarial_loss = configuration['alpha_adversarial_loss']
alpha_kl               = configuration['alpha_kl']
learning_rate          = configuration['learning_rate'] # 0.00005

model_path     = '/users/iris/rsourty/experiments/kdmkr/tableau_2/models_id'
dataset_path   = f'/users/iris/rsourty/experiments/kdmkr/kdmkr/datasets/{dataset_name}_tableau_2'

dict_teachers          = collections.OrderedDict()
dict_optimizers        = collections.OrderedDict()
dict_datasets          = collections.OrderedDict()
dict_negative_sampling = collections.OrderedDict()
dict_validation        = collections.OrderedDict()

for id_teacher in range(1, nb_teacher + 1):

    name_teacher = f'{id_teacher}_{nb_teacher}'

    dict_datasets[name_teacher] = stream.FetchDataset(
        train      = utils.read_csv(f'{dataset_path}/train_{name_teacher}.csv'),
        valid      = utils.read_csv(f'{dataset_path}/valid.csv'),
        test       = utils.read_csv(f'{dataset_path}/test.csv'),
        batch_size = batch_size,
        shuffle    = True,
        seed       = 42,
        entities   = utils.read_json(f'{dataset_path}/entities_{name_teacher}.json'),
        relations  = utils.read_json(f'{dataset_path}/relations.json'),
    )

    dict_teachers[name_teacher] = model.TransE(
        hidden_dim = hidden_dim,
        n_entity   = dict_datasets[name_teacher].n_entity,
        n_relation = dict_datasets[name_teacher].n_relation,
        gamma      = gamma,
    )

    dict_teachers[name_teacher] = dict_teachers[name_teacher].to(device)

    dict_optimizers[name_teacher] = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dict_teachers[name_teacher].parameters()), lr = learning_rate)

    dict_negative_sampling[name_teacher] = sampling.NegativeSampling(
        size       = negative_sampling_size,
        entities   = dict_datasets[name_teacher].entities,
        relations  = dict_datasets[name_teacher].relations,
        train_triples = (utils.read_csv(f'/users/iris/rsourty/experiments/kdmkr/kdmkr/datasets/{evaluation_folder}/train.csv')),
        seed = 42
    )

    dict_validation[name_teacher] = evaluation.Evaluation(
        all_true_triples = (
            utils.read_csv(f'/users/iris/rsourty/experiments/kdmkr/kdmkr/datasets/{evaluation_folder}/train.csv') +
            dict_datasets[name_teacher].test +
            dict_datasets[name_teacher].valid
        ),
        entities         = dict_datasets[name_teacher].entities,
        relations        = dict_datasets[name_teacher].relations,
        batch_size       = 2,
        device           = device,
    )

def get_dict_distillation(nb_teacher, dict_datasets, dict_teachers, batch_size_entity,
    batch_size_relation, n_random_entities, n_random_relations, device):

    dict_distillation = collections.OrderedDict()

    for id_teacher in range(1, nb_teacher + 1):

        for id_student in range(1, nb_teacher + 1):

            if id_teacher != id_student:

                name_teacher = f'{id_teacher}_{nb_teacher}'
                name_student = f'{id_student}_{nb_teacher}'

                dict_distillation[f'{name_teacher}_{name_student}'] = distillation.Distillation(
                    teacher_entities    = dict_datasets[name_teacher].entities,
                    teacher_relations   = dict_datasets[name_teacher].relations,
                    student_entities    = dict_datasets[name_student].entities,
                    student_relations   = dict_datasets[name_student].relations,
                    device              = device,
                    sampling            = distillation.TopKSampling(
                        teacher_entities    = dict_datasets[name_teacher].entities,
                        teacher_relations   = dict_datasets[name_teacher].relations,
                        student_entities    = dict_datasets[name_student].entities,
                        student_relations   = dict_datasets[name_student].relations,
                        teacher             = dict_teachers[name_teacher],
                        batch_size_entity   = batch_size_entity,
                        batch_size_relation = batch_size_relation,
                        n_random_entities   = n_random_entities,
                        n_random_relations  = n_random_relations,
                    )
                )

    return dict_distillation

dict_distillation = get_dict_distillation(
    nb_teacher          = nb_teacher,
    dict_datasets       = dict_datasets,
    dict_teachers       = dict_teachers,
    batch_size_entity   = batch_size_entity,
    batch_size_relation = batch_size_relation,
    n_random_entities   = n_random_entities,
    n_random_relations  = n_random_relations,
    device              = device
)

bar = tqdm.tqdm(range(1, max_step), position=0)

metrics = {name_teacher: stats.RollingMean(1000) for name_teacher in dict_teachers.keys()}

for step in bar:

    loss_teachers = {}
    dict_positive_samples = {}

    for name_teacher, dataset in dict_datasets.items():

        positive_sample, weight, mode = next(dataset)

        negative_sample = dict_negative_sampling[name_teacher].generate(
            positive_sample = positive_sample,
            mode            = mode,
        )

        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        weight          = weight.to(device)

        # Store positive sample for distillation
        dict_positive_samples[name_teacher] = positive_sample

        positive_score  = dict_teachers[name_teacher](positive_sample)
        negative_score  = dict_teachers[name_teacher](
            (positive_sample, negative_sample),
            mode = mode
        )

        loss_teachers[name_teacher] = loss.Adversarial()(
            positive_score = positive_score,
            negative_score = negative_score,
            weight         = weight,
            alpha          = alpha_adversarial_loss,
        ) * (1 - alpha_kl)

    for id_teacher in range(1, nb_teacher + 1):

        for id_student in range(1, nb_teacher + 1):

            if id_teacher != id_student:

                name_teacher = f'{id_teacher}_{nb_teacher}'
                name_student = f'{id_student}_{nb_teacher}'

                loss_distillation = dict_distillation[f'{name_teacher}_{name_student}'].distill(
                    teacher         = dict_teachers[name_teacher],
                    student         = dict_teachers[name_student],
                    positive_sample = dict_positive_samples[name_teacher]
                )

                if loss_distillation['head'] is not None:
                    loss_teachers[name_student] += loss_distillation['head'] * alpha_kl

                if loss_distillation['relation'] is not None:
                    loss_teachers[name_student] += loss_distillation['relation'] * alpha_kl

                if loss_distillation['tail'] is not None:
                    loss_teachers[name_student] += loss_distillation['tail'] * alpha_kl

    for name_teacher, _ in dict_datasets.items():

        loss_teachers[name_teacher].backward()

        dict_optimizers[name_teacher].step()

        dict_optimizers[name_teacher].zero_grad()

        metrics[name_teacher].update(loss_teachers[name_teacher].item())

    dict_distillation = get_dict_distillation(
        nb_teacher          = nb_teacher,
        dict_datasets       = dict_datasets,
        dict_teachers       = dict_teachers,
        batch_size_entity   = batch_size_entity,
        batch_size_relation = batch_size_relation,
        n_random_entities   = n_random_entities,
        n_random_relations  = n_random_relations,
        device              = device
    )

    if step % 5 == 0:
        bar.set_description(
            str({f'{name_teacher}: {metric.get():6f}' for name_teacher, metric in metrics.items()}))

    if step % 2000 == 0:

        for name_teacher, teacher in dict_teachers.items():

            teacher = teacher.eval()

            scores = dict_validation[name_teacher].eval(
                model   = teacher,
                dataset = dict_datasets[name_teacher].test
            )

            teacher = teacher.train()

            print(f'{name_teacher}: {scores}')

            # Set path HERE
            with open(f'/users/iris/rsourty/experiments/kdmkr/tableau_2/models/our_model_{dataset_name}_{name_teacher}_{scores}.pickle', 'wb') as handle:

                pickle.dump(teacher, handle, protocol = pickle.HIGHEST_PROTOCOL)
