<div align="center">
    <h1> Kdmkb </h1>
</div>
</br>

<div align="center">
    <img src="docs/img/pytorch.png" alt="pytorch" width="35%" vspace="20">
</div>
</br>

<p align="center">
  <code>kdmkb</code> is a library dedicated to <b>knowledge graph embeddings.</b> The purpose of this library is to provide modular tools using Pytorch.
Kdmkb provides datasets, models and tools to evaluate performance of models. Kdmkb makes possible to distil the knowledge of a model (teacher) to another model (student).</p>
</br>

## Table of contents

- [Table of contents](#table-of-contents)
- [👾 Installation](#-installation)
- [🗂 Datasets](#-datasets)
- [🤖 Models](#-models)
- [🚃 Training](#-training)
    - [✈️ Pipeline](#-pipeline)
    - [🔧 Lower level](#-lower-level)   
- [📊 Evaluation](#-evaluation)
- [🎁 Distillation](#-distillation)
- [🧰 Development](#-development)
- [🗒 License](#-license)

## 👾 Installation

You should be able to install and use this library with any Python version above 3.6.

```sh
$ pip install git+https://github.com/raphaelsty/kdmkb
$ cd kmkb
$ pip install -r requirements.txt
```

## 🗂 Datasets

**Datasets available:**

- `datasets.CountriesS1`
- `datasets.CountriesS2`
- `datasets.CountriesS3`
- `datasets.Fb13`
- `datasets.Fb15k`
- `datasets.Fb15k237`
- `datasets.Kinship`
- `datasets.Nations`
- `datasets.Nell995`
- `datasets.Umls`
- `datasets.Wn11`
- `datasets.Wn18`
- `datasets.Wn18rr`
- `datasets.Yago310`

`Wn18rr` and `Fb15k237` have textual mentions based on the work of [villmow](https://github.com/villmow/datasets_knowledge_embedding). 

**Load dataset:**

```python
from kdmkb import datasets

dataset = datasets.Wn18rr(batch_size=512, shuffle=True)

dataset
```

```python
Wn18rr dataset
    Batch size          512
    Entities            40923
    Relations           11
    Shuffle             True
    Train triples       86834
    Validation triples  3033
    Test triples        3134
```

**Load custom dataset:**

<b> You can build embeddings of your own knowledge base </b> using the `datasets.Fetch` module. It is necessary to provide the index of entities and relationships with an associated training set. Optionally, you can provide validation or test data to the dataset to validate your model later.

```python
from kdmkb import datasets

entities = {
   'bicycle'   : 0,
   'bike'      : 1,
   'car'       : 2,
   'truck'     : 3,
   'automobile': 4,
   'brand'     : 5,
}

relations = {
    'is_a' : 0,
    'not_a': 1,
}

train = [
    (0, 0, 2),
    (1, 0, 2),
    (2, 1, 3),
]

test = [
   (3, 1, 2),
   (4, 1, 5),
]

dataset = datasets.Fetch(
    train      = train, 
    test       = test, 
    entities   = entities, 
    relations  = relations,
    batch_size = 3, 
    shuffle    = True,
    seed       = 42
)

dataset

```

```python
Fetch dataset
    Batch size          3
    Entities            6
    Relations           2
    Shuffle             True
    Train triples       3
    Validation triples  0
    Test triples        2
```

## 🤖 Models

**Models available:**

- `models.TransE`
- `models.DistMult`
- `models.RotatE`
- `models.pRotatE`
- `models.ComplEx`

**Initialize a model:**

```python
from kdmkb import models

model = models.RotatE(
   n_entity   = dataset.n_entity, 
   n_relation = dataset.n_relation, 
   gamma      = 3, 
   hidden_dim = 500
)

model
```

```python
RotatE model
    Entities embeddings  dim  1000 
    Relations embeddings dim  500  
    Gamma                     3.0  
    Number of entities        40923 
    Number of relations       11   
```

**Set learning rate:**

```python
import torch

learning_rate = 0.00005

optimizer = torch.optim.Adam(
   filter(lambda p: p.requires_grad, model.parameters()),
   lr = learning_rate,
)

```

🤩 **Extract embeddings**

You can extract embeddings from entities and relationships computed by the model with the `models.embeddings` property.

```python
model.embeddings['entities']
```

```python
{0: tensor([ 0.7645,  0.8300, -0.2343]), 1: tensor([ 0.9186, -0.2191,  0.2018])}

```

```python
model.embeddings['relations']
```

```python
{0: tensor([-0.4869,  0.5873,  0.8815]), 1: tensor([-0.7336,  0.8692,  0.1872])}

```

## 🚃 Training

To train a model from `kdmkb` to the link prediction task you can copy and paste the code below. There are two alternatives for training a model. The first is to use a pipeline that takes care of the training loop for you. The second is to use `kdmkb` at a lower level.

#### ✈️ Pipeline:

Your can train our model using `compose.Pipeline`. It's a convenient class to make knowledge graph embeddings easily.You will simply have to select:

- A **dataset** or any available dataset in kbmkb. It is strongly recommended that the dataset has a validation or test set.

- A **model** and the associated hyper parameters.

- An **optimizer** from pytorch.

- A **sampling** method available in kdmkb such as `sampling.NegativeSampling`. This module allows you to generate negative samples from existing triplets. It does not generate triplets that are in the training set. You can also use your own `sampling` function.

- An **evaluation** process.

```python
from kdmkb import compose
from kdmkb import datasets
from kdmkb import evaluation
from kdmkb import losses
from kdmkb import models
from kdmkb import sampling
from kdmkb import utils

import torch

_ = torch.manual_seed(42)

device = 'cuda' # cpu if you do not own a gpu.

dataset  = datasets.Wn18rr(batch_size = 512, shuffle = True, seed = 42)

sampling = sampling.NegativeSampling(
       size          = 1024,
       train_triples = dataset.train,
       entities      = dataset.entities,
       relations     = dataset.relations,
       seed          = 42,
)

model = models.RotatE(
   n_entity   = dataset.n_entity,
   n_relation = dataset.n_relation,
   gamma      = 6,
   hidden_dim = 500,
)

model = model.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 0.00005,
)

evaluation = evaluation.Evaluation(
   true_triples = (
      dataset.train +
       dataset.valid + # Drop it if you do not have a valid set.
       dataset.test    # Drop it if you do not have a test set.
   ),
   entities   = dataset.entities,
   relations  = dataset.relations,
   batch_size = 8,
   device     = device,
)

pipeline = compose.Pipeline(
    max_step              = 80000, 
    eval_every            = 2000,  
    early_stopping_rounds = 3,
    device                = device,
)

pipeline = pipeline.learn(
    model      = model,
    dataset    = dataset,
    evaluation = evaluation,
    sampling   = sampling,
    optimizer  = optimizer,
    loss       = losses.Adversarial(alpha = 0.5),
)

utils.export_embeddings('./', dataset, model)
```

Output:

```python
Step: 2000. 
Validation scores - {'MRR':..., 'MR':..., 'HITS@1':..., 'HITS@3':..., 'HITS@10':...}
Test scores - {'MRR':..., 'MR':..., 'HITS@1':..., 'HITS@3':..., 'HITS@10':...}

Step: 4000. 
Validation scores - {'MRR':..., 'MR':..., 'HITS@1':..., 'HITS@3':..., 'HITS@10':...}
Test scores - {'MRR':..., 'MR':..., 'HITS@1':..., 'HITS@3':..., 'HITS@10':...}

...
```

#### 🔧 Lower level:

You can train a model at a lower level to control all the parameters of model training:

```python
from kdmkb import datasets
from kdmkb import losses 
from kdmkb import models
from kdmkb import sampling
from kdmkb import utils

from creme import stats

import torch

_ = torch.manual_seed(42)

device = 'cuda' # 'cpu' if do not own a gpu.

dataset = datasets.Wn18rr(batch_size=512, shuffle=True, seed=42)

negative_sampling = sampling.NegativeSampling(
   size          = 1024,
   train_triples = dataset.train,
   entities      = dataset.entities,
   relations     = dataset.relations,
   seed          = 42,
)

model = models.RotatE(
   n_entity   = dataset.n_entity, 
   n_relation = dataset.n_relation, 
   gamma      = 3, 
   hidden_dim = 500
)

model = model.to(device)

optimizer = torch.optim.Adam(
   filter(lambda p: p.requires_grad, model.parameters()),
   lr = 0.00005,
)

loss        = losses.Adversarial(alpha=0.5)
metric_loss = stats.RollingMean(1000)

bar = utils.Bar(step = 80000, update_every = 30)

for _ in bar():

    positive_sample, weight, mode=next(dataset)
    
    negative_sample = negative_sampling.generate(
        positive_sample = positive_sample,
        mode            = mode
    )
    
    positive_sample = positive_sample.to(device)
    negative_sample = negative_sample.to(device)
    weight = weight.to(device)
    
    positive_score = model(positive_sample)
    negative_score = model(negative_sample)
    
    error = loss(positive_score, negative_score, weight)
    
    error.backward()
    
    _ = optimizer.step()
    
    metric_loss.update(error.item())
    
    bar.set_description(f'loss: {metric_loss.get():4f}')
    

# Export embeddings as json file.
utils.export_embeddings('./', dataset, model)
```

## 📊 Evaluation

You can evaluate the performance of your models with the `evaluation` module. Evaluating *link prediction task* is available using `Evaluation.eval` and *relation prediction task* using `Evaluation.eval_relation`. By giving the training, validation and test triples to the `true_triples`parameter, you will calculate the `filtered` metrics. You can calculate the `raw` metrics by leaving `true_triples` to default value.

```python
from kdmkb import evaluation

validation = evaluation.Evaluation(
    true_triples = (
        dataset.train + 
        dataset.valid + 
        dataset.test
    ),
    entities   = dataset.entities, 
    relations  = dataset.relations, 
    batch_size = 8,
    device     = device,
)

validation.eval(model = model, dataset = dataset.valid)
```

```python
{'MRR': 0.5833, 'MR': 400.0, 'HITS@1': 20.25, 'HITS@3': 30.0, 'HITS@10': 40.0}
```

```python
validation.eval(model = model, dataset = dataset.test)
```

```python
{'MRR': 0.5833, 'MR': 600.0, 'HITS@1': 21.35, 'HITS@3': 38.0, 'HITS@10': 41.0}

```

```python
validation.eval_relations(model=model, dataset=dataset.test)

```

```python
{'MRR_relations': 1.0, 'MR_relations': 1.0, 'HITS@1_relations': 1.0, 'HITS@3_relations': 1.0, 'HITS@10_relations': 1.0}
```

## 🎁 Distillation

You can distil the knowledge of a pre-trained model. Distillation allows a model to reproduce the results of a pre-trained model. In some configurations, the student can overtake the master. The teacher and student must have a `distill` class method as defined in `kdmkb`.

```python
from kdmkb import datasets
from kdmkb import distillation
from kdmkb import models 

import torch

_ = torch.manual_seed(42)

device = 'cuda' # 'cpu' if do not own a gpu.

dataset = datasets.Wn18rr(
    batch_size = 3, 
    shuffle    = True, 
    seed       = 42
)

teacher = # Load pre-trained model

teacher = teacher.to(device) 

student = models.RotatE(
   n_entity   = dataset.n_entity, 
   n_relation = dataset.n_relation, 
   gamma      = 3, 
   hidden_dim = 500
)

student = student.to(device)

optimizer = torch.optim.Adam(
   filter(lambda p: p.requires_grad, student.parameters()),
   lr = 0.00005,
)

# Initialize distillation process:
distillation = distillation.Distillation(
    teacher_entities  = dataset.entities,
    student_entities  = dataset.entities,
    teacher_relations = dataset.relations,
    student_relations = dataset.relations,
    sampling          = distillation.UniformSampling( # Top K Soon
        batch_size_entity   = 20,
        batch_size_relation = 11,
        seed                = 42,
    ),
    device = device,
)

for _ in range(20000):

    positive_sample, weight, mode = next(dataset)
    
    positive_sample = positive_sample.to(device)
    weight = weight.to(device)
    
    loss = distillation.distill(
        teacher = teacher,
        student = student,
        positive_sample = positive_sample,
    )
    
    loss.backward()
    
    _ = optimizer.step()

```


## 🧰 Development

```sh
# Download and navigate to the source code
$ git clone https://github.com/raphaelsty/kdmkb
$ cd kmkb

# Create a virtual environment
$ python3 -m venv env
$ source env/bin/activate

# Install 
$ pip install -r requirements.txt
$ python setup.py install 

# Run tests
$ python -m pytest
```

## 🗒 License

This project is free and open-source software licensed under the [MIT license](https://github.com/raphaelsty/river/blob/master/LICENSE).
