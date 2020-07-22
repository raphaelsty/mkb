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
- [📊 Evaluation](#-evaluation)
- [🤩 Embeddings](#-embeddings)
- [🧰 Development](#-development)
- [🗒 License](#-license)

## 👾 Installation

You should be able to install and use this library with any Python version above 3.6.

```sh
$ pip install git+https://github.com/raphaelsty/kdmkb
```

## 🗂 Datasets

- **WN18RR**

```python
>>> from kdmkb import datasets

>>> dataset = Wn18rr(batch_size=512, shuffle=True)

>>> dataset
Wn18rr dataset
    Batch size          512
    Entities            40923
    Relations           11
    Shuffle             True
    Train triples       86834
    Validation triples  3033
    Test triples        3134

```

- **FB15K237**

```python
>>> from kdmkb import datasets

>>> dataset = Fb15k237(batch_size=512, shuffle=True)

>>> dataset
Fb15k237 dataset
    Batch size          512
    Entities            14541
    Relations           237
    Shuffle             True
    Train triples       272115
    Validation triples  17535
    Test triples        20466

```

- **Using custom dataset:**

```python
>>> from kdmkb import datasets

>>> entities = {
...    0: 'bicycle',
...    1: 'bike',
...    2: 'car',
...    3: 'truck',
...    4: 'automobile',
...    5: 'brand',
... }

>>> relations = {
...     0: 'is_a',
...     1: 'not_a',
... }

>>> train = [
...     (0, 0, 2),
...     (1, 0, 2),
...     (2, 1, 3),
... ]

>>> test = [
...    (3, 1, 2),
...    (4, 1, 5),
... ]

>>> dataset = datasets.Fetch(
...     train      = train, 
...     test       = test, 
...     entities   = entities, 
...     relations  = relations,
...     batch_size = 3, 
...     seed       = 42
... )

>>> dataset
Fetch dataset
    Batch size          3
    Entities            6
    Relations           2
    Shuffle             False
    Train triples       3
    Validation triples  0
    Test triples        2


```

## 🤖 Models

- **TransE**

```python
>>> from kdmkb import models

>>> model = models.TransE(
...    n_entity   = dataset.n_entity, 
...    n_relation = dataset.n_relation, 
...    gamma.     = 3, 
...    hiddem_dim = 500
... )

```

- **DistMult**

```python
>>> from kdmkb import models

>>> model = models.DistMult(
...    n_entity   = dataset.n_entity, 
...    n_relation = dataset.n_relation, 
...    gamma.     = 3, 
...    hiddem_dim = 500
... )
```

- **RotatE**

```python
>>> from kdmkb import models

>>> model = models.RotatE(
...    n_entity   = dataset.n_entity, 
...    n_relation = dataset.n_relation, 
...    gamma.     = 3, 
...    hiddem_dim = 500
... )

```

- **ProtatE**

```python
>>> from kdmkb import models

>>> model = models.ProtatE(
...    n_entity   = dataset.n_entity, 
...    n_relation = dataset.n_relation, 
...    gamma.     = 3, 
...    hiddem_dim = 500
... )

```

- **ComplEx**

```python
from kdmkb import models

>>> model = models.ComplEx(
...    n_entity   = dataset.n_entity, 
...    n_relation = dataset.n_relation, 
...    gamma.     = 3, 
...    hiddem_dim = 500
... )

```

## 🚃 Training

```python
>>> from kdmkb import datasets
>>> from kdmkb import losses 
>>> from kdmkb import models
>>> from kdmkb import sampling

>>> import torch

>>> _ = torch.manual_seed(42)

>>> device = 'cuda' # 'cpu' if you do not own a gpu.

>>> dataset = Wn18rr(batch_size=512, shuffle=True, seed=42)

>>> negative_sampling = sampling.NegativeSampling(
...    size = 2,
...    train_triples = dataset.train,
...    entities = dataset.entities,
...    relations = dataset.relations,
...    seed = 42,
... )

>>> model = models.RotatE(
...    n_entity   = dataset.n_entity, 
...    n_relation = dataset.n_relation, 
...    gamma.     = 3, 
...    hiddem_dim = 500
... )

>>> model = model.to(device)

>>> optimizer = torch.optim.Adam(
...    filter(lambda p: p.requires_grad, rotate.parameters()),
...    lr = 0.00005,
... )

>>> loss = losses.Adversarial()

>>> for _ in range(80000):
...     positive_sample, weight, mode=next(dataset)
...     positive_score = rotate(positive_sample)
...     negative_sample = negative_sampling.generate(
...         positive_sample = positive_sample,
...         mode            = mode
...     )
...     negative_score = rotate(
...         (positive_sample, negative_sample), 
...         mode=mode
...     )
...     loss(positive_score, negative_score, weight, alpha=0.5).backward()
...     _ = optimizer.step()

```

## 📊 Evaluation

You can evaluate the performance of your models with the `evaluation` module. 

```python
>>> from kdmkb import evaluation

>>> validation = evaluation.Evaluation(
...     true_triples = (
...         dataset.train + 
...         dataset.valid + 
...         dataset.test
...     ),
...     entities   = dataset.entities, 
...     relations  = dataset.relations, 
...     batch_size = 8,
...     device     = device,
... )

>>> validation.eval(model = model, dataset = dataset.valid)
{'MRR': 0.5833, 'MR': 400.0, 'HITS@1': 20.25, 'HITS@3': 30.0, 'HITS@10': 40.0}

>>> validation.eval(model = model, dataset = dataset.test)
{'MRR': 0.5833, 'MR': 600.0, 'HITS@1': 21.35, 'HITS@3': 38.0, 'HITS@10': 41.0}

```

## 🤩 Embeddings

```python
>>> model.embeddings['entities']
{0: tensor([ 0.7645,  0.8300, -0.2343]), 1: tensor([ 0.9186, -0.2191,  0.2018])}

>>> model.embeddings['relations']
{0: tensor([-0.4869,  0.5873,  0.8815]), 1: tensor([-0.7336,  0.8692,  0.1872])}
        
```

## 🧰 Development

```sh
# Download and navigate to the source code
$ git clone hhttps://github.com/raphaelsty/kdmkb
$ cd kmkb

# Create a virtual environment
$ python3 -m venv env
$ source env/bin/activate

# Install in development mode
$ pip install -e ".[dev]"
$ python setup.py develop

# Run tests
$ python -m pytest
```

## 🗒 License

This project is free and open-source software licensed under the [MIT license](https://github.com/raphaelsty/river/blob/master/LICENSE).
