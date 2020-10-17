<div align="center">
    <img src="docs/img/logo.png" alt="PyTorch" width="45%" vspace="50">
</div>
</br>

<div align="center">
    <img src="docs/img/Pytorch.png" alt="PyTorch" width="35%" vspace="20">
</div>
</br>

<p align="center">
  <code>mkb</code> is a library dedicated to <b>knowledge graph embeddings.</b> The purpose of this library is to provide modular tools using PyTorch.
mkb provides datasets, models and tools to evaluate performance.</p>
</br>

## 💬 Citations 

The **Mkb** library was developed for the research paper **Knowledge Base Embedding By Cooperative Knowledge Distillation** soon to be published at **Coling2020**.

**Sourty, Raphaël and G. Moreno, Jose and Servant, François-Paul and Tamine-Lechani, Lynda.**


## Table of contents

- [Table of contents](#table-of-contents)
- [👾 Installation](#-installation)
- [⚡️ Quickstart](#-quickstart)
- [🗂 Datasets](#-datasets)
- [🤖 Models](#-models)
- [🎭 Negative sampling](#-negative-sampling)
- [🤖 Train your model](#-train-you-model)  
- [📊 Evaluation](#-evaluation)
- [🤩 Get embeddings](#-get-embeddings)
- [🎁 Distillation](#-distillation)
- [🧰 Development](#-development)
- [🗒 License](#-license)

## 👾 Installation

You should be able to install and use this library with any Python version above 3.6.

```sh
$ pip install git+https://github.com/raphaelsty/mkb
$ cd kmkb
$ pip install -r requirements.txt
```

 

## ⚡️ Quickstart:

Load or initialize your dataset as a list of triplets:

```python
train = [    
    ('🦆', 'is a', 'bird'),
    ('🦅', 'is a', 'bird'),
    
    ('🦆', 'lives in', '🌳'),
    ('🦉', 'lives in', '🌳'),
    ('🦅', 'lives in', '🏔'),
    
    ('🦉', 'hability', 'fly'),
    ('🦅', 'hability', 'fly'),
    
    ('🐌', 'is a', 'mollusc'),
    ('🐜', 'is a', 'insect'),
    ('🐝', 'is a', 'insect'),
    
    ('🐌', 'lives in', '🌳'),
    ('🐝', 'lives in', '🌳'),
    
    ('🐝', 'hability', 'fly'),
    
    ('🐻', 'is a', 'mammal'),
    ('🐶', 'is a', 'mammal'),
    ('🐨', 'is a', 'mammal'),
    
    ('🐻', 'lives in', '🏔'),
    ('🐶', 'lives in', '🏠'),
    ('🐱', 'lives in', '🏠'),
    ('🐨', 'lives in', '🌳'),
    
    ('🐬', 'lives in', '🌊'),
    ('🐳', 'lives in', '🌊'),
    
    ('🐋', 'is a', 'marine mammal'),
    ('🐳', 'is a', 'marine mammal'),
]

valid = [
    ('🦆', 'hability', 'fly'),
    ('🐱', 'is_a', 'mammal'),
    ('🐜', 'lives_in', '🌳'),
    ('🐬', 'is_a', 'marine mammal'),
    ('🐋', 'lives_in', '🌊'),
    ('🦉', 'is a', 'bird'),
]
```

Train your model to make coherent embeddings for each entities and relations of your dataset using a pipeline:

```python
from mkb import datasets
from mkb import models
from mkb import losses
from mkb import sampling
from mkb import evaluation
from mkb import compose

import torch

_ = torch.manual_seed(42)

# Set device = 'cuda' if you own a gpu.
device = 'cpu' 

dataset = datasets.Dataset(
    train      = train,
    valid      = valid,
    batch_size = 24,
)

model = models.RotatE(
    entities   = dataset.entities,
    relations  = dataset.relations,
    gamma      = 3,
    hidden_dim = 200,
)

model = model.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 0.003,
)

negative_sampling = sampling.NegativeSampling(
    size          = 24,
    train_triples = dataset.train,
    entities      = dataset.entities,
    relations     = dataset.relations,
    seed          = 42,
)

validation = evaluation.Evaluation(
    true_triples = dataset.true_triples,
    entities     = dataset.entities,
    relations    = dataset.relations,
    batch_size   = 8,
    device       = device,
)

pipeline = compose.Pipeline(
    epochs                = 100,
    eval_every            = 50,
    early_stopping_rounds = 3,
    device                = device,
)

pipeline = pipeline.learn(
    model      = model,
    dataset    = dataset,
    evaluation = validation,
    sampling   = negative_sampling,
    optimizer  = optimizer,
    loss       = losses.Adversarial(alpha=1)
)
```
<br>

<details><summary>**Plot embeddings:**</summary>

```python
from sklearn import manifold

from sklearn import cluster

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

emojis_tokens = {
    '🦆': 'duck', 
    '🦅': 'eagle', 
    '🦉': 'owl', 
    '🐌': 'snail',
    '🐜': 'ant', 
    '🐝': 'bee', 
    '🐻': 'bear', 
    '🐶': 'dog', 
    '🐨': 'koala', 
    '🐱': 'cat', 
    '🐬': 'dolphin', 
    '🐳': 'whale', 
    '🐋': 'humpback whale', 
}

embeddings = pd.DataFrame(model.embeddings['entities']).T.reset_index()

embeddings = embeddings[embeddings['index'].isin(emojis_tokens)].set_index('index')

tsne = manifold.TSNE(n_components = 2, random_state = 42, n_iter=1500, perplexity=3, early_exaggeration=100)

kmeans = cluster.KMeans(n_clusters = 5, random_state=42)

X = tsne.fit_transform(embeddings)
X = pd.DataFrame(X, columns = ['dim_1', 'dim_2'])
X['cluster'] = kmeans.fit_predict(X)

%config InlineBackend.figure_format = 'retina'

fgrid = sns.lmplot(
    data = X, 
    x = 'dim_1', 
    y = 'dim_2', 
    hue = 'cluster', 
    fit_reg = False, 
    legend = False, 
    legend_out = False,
    height = 7, 
    aspect = 1.6,
    scatter_kws={"s": 500}
)

ax = fgrid.axes[0,0]
ax.set_ylabel('')    
ax.set_xlabel('')
ax.set(xticklabels=[]) 
ax.set(yticklabels=[]) 

for i, label in enumerate(embeddings.index):
    
     ax.text(
         X['dim_1'][i] + 1, 
         X['dim_2'][i], 
         emojis_tokens[label], 
         horizontalalignment = 'left', 
         size = 'medium', 
         color = 'black', 
         weight = 'semibold',
     )
        
plt.show()
```

</details>

<div align="center">
    <img src="docs/img/plot.png" alt="PyTorch" width="100%" vspace="20">
</div>
</br>

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

**Load existing dataset:**

```python
from mkb import datasets

dataset = datasets.Wn18rr(batch_size=256)

dataset
```

```python
Wn18rr dataset
    Batch size          256
    Entities            40923
    Relations           11
    Shuffle             True
    Train triples       86834
    Validation triples  3033
    Test triples        3134
```

**Or create your own dataset:**

```python
from mkb import datasets

train = [
    ('🦆', 'is a', 'bird'),
    ('🦅', 'is a', 'bird'),
    ('🦉', 'hability', 'fly'),
    ('🦅', 'hability', 'fly')
]

valid = [
    ('🦉', 'is a', 'bird')
]

test = [
    ('🦆', 'hability', 'fly')
]

dataset = datasets.Dataset(
    train = train,
    valid = valid,
    test = test,
    batch_size = 3,
    seed = 42,
)

dataset
```

```python
Dataset dataset
        Batch size  3   
          Entities  5   
         Relations  2   
           Shuffle  True
     Train triples  4   
Validation triples  1   
      Test triples  1
```

## 🤖 Models

Knowledge graph models build latent representations of nodes (entities) and relationships in the graph. These models implement an optimization process to represent the entities and relations in a consistent space.

**Models available:**

- `models.TransE`
- `models.DistMult`
- `models.RotatE`
- `models.pRotatE`
- `models.ComplEx`
- `models.ConvE` # A notebook will come soon to train ConvE.

**Initialize a model:**

```python
from mkb import models

model = models.RotatE(
   n_entity   = dataset.entities, 
   n_relation = dataset.relations, 
   gamma      = 6, 
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

**Set the learning rate of the model:**

```python
import torch

learning_rate = 0.00005

optimizer = torch.optim.Adam(
   filter(lambda p: p.requires_grad, model.parameters()),
   lr = learning_rate,
)

```

## 🎭 Negative sampling 

Knowledge graph embedding models learn to distinguish existing triplets from generated triplets. The `sampling` module allows to generate triplets that do not exist in the dataset. 

```python
from mkb import sampling

negative_sampling = sampling.NegativeSampling(
    size          = 256,
    train_triples = dataset.train,
    entities      = dataset.entities,
    relations     = dataset.relations,
    seed          = 42,
)
```

## 🤖 Train your model 

You can train your model using a pipeline:

```python
pipeline = compose.Pipeline(
    epochs                = 100,
    eval_every            = 50,
    early_stopping_rounds = 3,
    device                = device,
)

pipeline = pipeline.learn(
    model      = model,
    dataset    = dataset,
    evaluation = validation,
    sampling   = negative_sampling,
    optimizer  = optimizer,
    loss       = losses.Adversarial(alpha=1)
)
```

You can also train your model with a lower level of abstraction:

```python
loss = losses.Adversarial(alpha=0.5)

for epoch in range(2000):
    
    for data in dataset:
        
        sample = data['sample'].to(device)
        weight = data['weight'].to(device)
        mode = data['mode']
        
        negative_sample = negative_sampling.generate(sample=sample, mode=mode)

        negative_sample = negative_sample.to(device)
        
        positive_score = model(sample)
        
        negative_score = model(
            sample=sample,
            negative_sample=negative_sample,
            mode=mode
        )
        
        error = loss(positive_score, negative_score, weight)
        
        error.backward()

        _ = optimizer.step()

        optimizer.zero_grad()
    
    validation_scores = validation.eval(dataset=dataset.valid, model=model)
    
    print(validation_scores)
```


## 📊 Evaluation

You can evaluate the performance of your models with the `evaluation` module. 

```python
from mkb import evaluation

validation = evaluation.Evaluation(
    true_triples = dataset.true_triples, 
    entities   = dataset.entities, 
    relations  = dataset.relations, 
    batch_size = 8,
    device     = device,
)
```

#### 🎯 Link prediction task:

The task of link prediction aim at finding the most likely head or tail for a given tuple. For example, the model should retrieve the entity `United States` for the triplet `('Barack Obama', 'president_of', ?)`.

Validate the model on the validation set:

```python
validation.eval(model = model, dataset = dataset.valid)

```

```python
{'MRR': 0.5833, 'MR': 400.0, 'HITS@1': 20.25, 'HITS@3': 30.0, 'HITS@10': 40.0}
```
Validate the model on the test set:

```python
validation.eval(model = model, dataset = dataset.test)
```

```python
{'MRR': 0.5833, 'MR': 600.0, 'HITS@1': 21.35, 'HITS@3': 38.0, 'HITS@10': 41.0}

```

#### 🔎 Link prediction detailed evaluation:

You can get a more detailed evaluation of the link prediction task and measure the performance of the model according to the type of relationship.

```python
validation.detail_eval(model=model, dataset=dataset.test, treshold=1.5)
```

```python
          head                               tail
          MRR   MR HITS@1 HITS@3 HITS@10     MRR   MR HITS@1 HITS@3 HITS@10
relation
1_1       0.5  2.0    0.0    1.0     1.0  0.3333  3.0    0.0    1.0     1.0
1_M       1.0  1.0    1.0    1.0     1.0  0.5000  2.0    0.0    1.0     1.0
M_1       0.0  0.0    0.0    0.0     0.0  0.0000  0.0    0.0    0.0     0.0
M_M       0.0  0.0    0.0    0.0     0.0  0.0000  0.0    0.0    0.0     0.0
  
```

#### ➡️ Relation prediction:

The task of relation prediction is to find the most likely relation for a given tuple (head, tail).

```python
validation.eval_relations(model=model, dataset=dataset.test)

```

```python
{'MRR_relations': 1.0, 'MR_relations': 1.0, 'HITS@1_relations': 1.0, 'HITS@3_relations': 1.0, 'HITS@10_relations': 1.0}
```

#### 🦾 Triplet classification

The triplet classification task is designed to predict whether or not a triplet exists. The triplet classification task is available for every datasets in `mkb` except `Countries` datasets.

```python
from mkb import evaluation

evaluation.find_treshold(
    model = model,
    X = dataset.classification_valid['X'],
    y = dataset.classification_valid['y'],
    batch_size = 10,
)

```

Best treshold found from triplet classification valid set and associated accuracy:

```python
{'threshold': 1.924787, 'accuracy': 0.803803}
```

```python
evaluation.accuracy(
    model = model,
    X = dataset.classification_test['X'],
    y = dataset.classification_test['y'],
    threshold = 1.924787,
    batch_size = 10,
)
```

Accuracy of the model on the triplet classification test set:

```python
0.793803
```

## 🤩 Get embeddings

You can extract embeddings from entities and relationships computed by the model with the `models.embeddings` property.

```python
model.embeddings['entities']
```

```python
{'hello': tensor([ 0.7645,  0.8300, -0.2343]), 'world': tensor([ 0.9186, -0.2191,  0.2018])}

```

```python
model.embeddings['relations']
```

```python
{'lorem': tensor([-0.4869,  0.5873,  0.8815]), 'ipsum': tensor([-0.7336,  0.8692,  0.1872])}

```

## 🎁 Distillation

The module `mkb.distillation` provides tools for distilling knowledge of knowledge graph embeddings models. Dedicated notebook will soon be available.

## 🧰 Development

```sh
# Download and navigate to the source code
$ git clone https://github.com/raphaelsty/mkb
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
