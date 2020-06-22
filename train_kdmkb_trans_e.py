from creme import stats

from kdmkr import distillation
from kdmkr import evaluation
from kdmkr import model
from kdmkr import loss
from kdmkr import sampling
from kdmkr import stream
from kdmkr import utils

import torch
import tqdm
import collections
import pickle

import sys
import yaml

device = 'cuda'

configuration_file = sys.argv[2]

with open(f'{configuration_file}') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    configuration = yaml.load(file, Loader=yaml.FullLoader)

percent_entities = configuration['percent_entities']
alpha_kl         = configuration['alpha_kl']

_ = torch.manual_seed(42)

path_data = '/users/iris/rsourty/experiments/kdmkr/kdmkr/datasets/kdmkb_datasets'

batch_size              = {'wn18rr': 256, 'fb15k237': 512 }
batch_size_intersection = {'wn18rr': 198, 'fb15k237': 184 }
hidden_dim              = {'wn18rr': 500, 'fb15k237': 1000}
gamma                   = {'wn18rr': 6  , 'fb15k237': 9   }
negative_sampling_size  = {'wn18rr': 512, 'fb15k237': 128 }
alpha_adversarial_loss  = {'wn18rr': 0.5, 'fb15k237': 1   }
device                  = 'cuda'
max_step                = 80000

dict_datasets          = collections.OrderedDict()
dict_validation        = collections.OrderedDict()
dict_intersections     = collections.OrderedDict()
dict_models            = collections.OrderedDict()
dict_optimizers        = collections.OrderedDict()
dict_negative_sampling = collections.OrderedDict()
dict_entities_intersection = collections.OrderedDict()

for kb in ['wn18rr', 'fb15k237']:

    dict_datasets[kb] = stream.FetchDataset(
        train      = utils.read_csv(file_path = f'{path_data}/train_{kb}.csv'),
        valid      = utils.read_csv(file_path = f'{path_data}/valid_{kb}.csv'),
        test       = utils.read_csv(file_path = f'{path_data}/test_{kb}.csv'),
        entities   = utils.read_json(file_path = f'{path_data}/entities_{kb}_{percent_entities}.json'),
        relations  = utils.read_json(file_path = f'{path_data}/relations_{kb}.json'),
        batch_size = batch_size[kb],
        seed       = 42,
    )

    dict_negative_sampling[kb] = sampling.NegativeSampling(
        size       = negative_sampling_size[kb],
        entities   = dict_datasets[kb].entities,
        relations  = dict_datasets[kb].relations,
        train_triples = (dict_datasets[kb].train),
        seed = 42
    )

    dict_validation[kb] = evaluation.Evaluation(
        all_true_triples = (
            dict_datasets[kb].train +
            dict_datasets[kb].valid +
            dict_datasets[kb].test
        ),
        entities   = dict_datasets[kb].entities,
        relations  = dict_datasets[kb].relations,
        batch_size = 2,
        device     = device,
    )

    dict_intersections[kb] = stream.FetchDataset(
        train      = utils.read_csv(file_path = f'{path_data}/intersection_{kb}_{percent_entities}.csv'),
        entities   = dict_datasets[kb].entities,
        relations  = dict_datasets[kb].relations,
        batch_size = batch_size_intersection[kb],
    )

    dict_models[kb] = model.TransE(
        n_entity   = dict_datasets[kb].n_entity,
        n_relation = dict_datasets[kb].n_relation,
        hidden_dim = hidden_dim[kb],
        gamma      = gamma[kb],
    )

    dict_models[kb] = dict_models[kb].to(device)

    dict_optimizers[kb] = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dict_models[kb].parameters()), lr = 0.00005)

    dict_entities_intersection[kb] = utils.read_json(
        file_path = f'{path_data}/entities_intersection_{kb}_{percent_entities}.json'
    )

def get_dict_distillation(dict_datasets, dict_models, device, dict_entities_intersection):

    dict_distillation = collections.OrderedDict()

    batch_size_entity = 20
    n_random_entities = 20

    if len(dict_entities_intersection) < batch_size_entity:
        batch_size_entity = len(dict_entities_intersection)
        n_random_entities = 0

    for teacher in ['wn18rr', 'fb15k237']:

        for student in ['wn18rr', 'fb15k237']:

            if teacher != student:

                dict_distillation[f'{teacher}_{student}'] = distillation.Distillation(
                    teacher_entities  = dict_entities_intersection[teacher],
                    teacher_relations = dict_datasets[teacher].relations,
                    student_entities  = dict_entities_intersection[student],
                    student_relations = dict_datasets[student].relations,
                    device            = device,
                    sampling          = distillation.TopKSampling(
                        teacher_entities  = dict_entities_intersection[teacher],
                        teacher_relations = dict_datasets[teacher].relations,
                        student_entities  = dict_entities_intersection[student],
                        student_relations = dict_datasets[student].relations,
                        teacher             = dict_models[teacher],
                        batch_size_entity   = batch_size_entity,
                        batch_size_relation = 1,
                        n_random_entities   = n_random_entities,
                        n_random_relations  = 0,
                        seed                = 42,
                    ),
                )
    return dict_distillation

dict_distillation = get_dict_distillation(dict_datasets, dict_models, device, dict_entities_intersection)

bar     = tqdm.tqdm(range(1, max_step), position=0)
metrics = collections.OrderedDict({kg: stats.RollingMean(1000) for kg in ['wn18rr', 'fb15k237']})

for step in bar:

    loss_models = {}

    for kb, dataset in dict_datasets.items():

        positive_sample, weight, mode = next(dataset)

        negative_sample = dict_negative_sampling[kb].generate(
            positive_sample = positive_sample,
            mode            = mode,
        )
        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        weight          = weight.to(device)

        positive_score  = dict_models[kb](positive_sample)
        negative_score  = dict_models[kb]((positive_sample, negative_sample), mode = mode)

        loss_models[kb] = loss.Adversarial()(
            positive_score = positive_score,
            negative_score = negative_score,
            weight         = weight,
            alpha          = alpha_adversarial_loss[kb],
        ) * (1 - alpha_kl)

    for teacher in ['wn18rr', 'fb15k237']:

        for student in ['wn18rr', 'fb15k237']:

            if teacher != student:

                distillation_sample, _, _ = next(dict_intersections[teacher])

                distillation_sample = distillation_sample.to(device)

                loss_distillation = dict_distillation[f'{teacher}_{student}'].distill(
                    teacher         = dict_models[teacher],
                    student         = dict_models[student],
                    positive_sample = distillation_sample,
                )

                if loss_distillation['head'] is not None:
                    loss_models[student] += loss_distillation['head'] * alpha_kl

                if loss_distillation['relation'] is not None:
                    loss_models[student] += loss_distillation['relation'] * alpha_kl

                if loss_distillation['tail'] is not None:
                    loss_models[student] += loss_distillation['tail'] * alpha_kl

    for kg in ['wn18rr', 'fb15k237']:

        loss_models[kg].backward()

        dict_optimizers[kg].step()

        dict_optimizers[kg].zero_grad()

        metrics[kg].update(loss_models[kg].item())

    dict_distillation = get_dict_distillation(dict_datasets, dict_models, device, dict_entities_intersection)

    if step % 5 == 0:

        bar.set_description(str({f'{kg}: {metric.get():6f}' for kg, metric in metrics.items()}))

    if step % 2000 == 0:

        for kg in ['wn18rr', 'fb15k237']:

            dict_models[kg] = dict_models[kg].eval()

            scores = dict_validation[kg].eval(
                model   = dict_models[kg],
                dataset = dict_datasets[kg].test
            )

            dict_models[kg] = dict_models[kg].train()

            print(f'{kg}: {scores}')

            # Set path HERE
            with open(f'/users/iris/rsourty/experiments/kdmkr/tableau_2/models/kdmkb_true_{kg}_{percent_entities}_{scores}.pickle', 'wb') as handle:

                pickle.dump(dict_models[kg], handle, protocol = pickle.HIGHEST_PROTOCOL)
