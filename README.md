#### KD-MKDR


Installation:
```
pip install -r requirements.txt
python setup.py install
```

---

#### Experiment 1

Join fb15k237 with Wordnet18rr. 

#### Target

Improving RotatE results on the dataset fb15k237 and vice-versa.

#### Realization
To identify common entities between the two KGs, I find triplets that have a common head and tail between the two KG. I make the hypothesize that the triplets that have a common head and tail in both KG contain common entities. I identify 464 triplets with identical head and tail that have the relation ``_has_part`` in wn18rr and ``location/location/contains`` in fb15k237. I therefore choose to replace the relation "location/location/contains" in fb15k237 by the relation "_has_part" in wn18rr. These are the only two relationships I can bring together. 


#### Results

I built a dataset common to wn18rr and fb15k237 containing 638 entities and a single relationship ``_has_part``. This dataset is composed of 1234 triplets. It contains mainly geographical information. 56.08% of triplets comes from fb15k237 and 43.92% comes from wn18rr. 

#### Discussion

The granularity of the relations of fb15k237 is finer than the one of wn18rr. This experiment does not allow to establish many links between the two KGs. 20.77% of the triplets in the validation set of fb15k237 contain at least one entity common to both KGs.  Only 0.7% of the triplets in the fb15K237 validation set contain two entities common to both KGs. 20.96% of the triplets in the fb15k237 test set contain at least one entity common to both KGs. Only 0.6% of the triplets in the fb15K237 test set contain two entities common to both KGs. 3.89% of the triplets in the wn18rr validation set contain at least one entity common to both KGs.  Only 0.5% of the triplets in the wn18rr validation set contain two entities common to both KGs. 3.98% of the triplets in the wn18rr test set contain at least one entity common to both KGs. Only 0.6% of the triplets in the wn18rr test set contain two entities common to both KGs.

#### Perspectives

I think the dataset I created may have bigger impact on fb15k237. Are my models already trained on wn18rr / fb15k237 experts on this new dataset? How much can be gained?

---

#### Experiment 2

What is the performance of a model trained on WN18RR on the dataset of experiment 1? What is the performance of a model trained on FB15K237 on the experiment 1 dataset? 

#### Target 

Identify the potential gain of a model trained on FB15K237 / WN18RR via distillation.

#### Realization

I selected two independently trained models on the WN18RR and FB15K237 datasets and measured their performance on the intersection of WN18RR and FB15K237 (experiment 1).

#### Results

Score of the RotatE model trained on WN18RR:

```HITS@10: 0.773501, HITS@1: 0.467180, HITS@3: 0.598055, MR: 461.707455, MRR: 0.563140```


Score of the RotatE model trained on FB15K237: 

```HITS@10: 0.713128, HITS@1: 0.357780, HITS@3: 0.577796, MR: 56.786467, MRR: 0.488970```

#### Discussion:

The model trained on WN18RR is better than the model trained on FB15K237 to find the missing entity in a tuple (integer, relation) among the triplets of the intersection of the two datasets WN18RR and FB15K237. One can suppose that the Knwoledge Graph WN18RR allows to better represent in a latent space the geographical entities.


#### Perspectives 

It would be interesting to identify the entities that the models classify correctly. We might get better results with distillation if the models provide different results.

---

#### Experiment 3

#### Target

L'objectif de cet expérience est de quantifier la performance de la distillation d'un étudiant vers plusieurs professeurs pour l'article kdmkb en utilisant les datasets WN18RR et FB15K237.


#### Realization

Pour réaliser cette expérience, je découpe WN18RR en deux, puis en trois et finalement en quatre partie de taille identique. Je répète l'opération avec FB15K237. 

- Independant: J'entraine un modèle sur chaque sous-ensemble des knowledges bases, on appel ces modèles "Independant". Best correspond au meilleur modèle 

- Xdistill~X: J'entraine ensuite un modèle par distillation. 

- J'entraine un modèle par distillation sur l'ensemble des modèles entrainé sur chacun sur 




| Dataset 	| Parts 	| Independant 	|  	| Xdistill~X 	|  	| Our model 	|  	|
|:--------:	|:-----:	|:-----------:	|:-----:	|:----------:	|:-----:	|:---------:	|:-----:	|
|  	|  	| Best 	| Worst 	| Best 	| Worst 	| Best 	| Worst 	|
| WN18RR 	|          2          	|        18.29      	|        18.13      	|       17.89      	|       17.54      	|  	|  	|
| WN18RR 	|               3               	|     12.09   	|    10.81  	|     12.39   	|    10.37  	|  	|  	|
| WN18RR 	|                    4                    	|  	|  	|  	|  	|  	|  	|
| FB15K237 	|          2          	|  	|  	|  	|  	|  	|  	|
| FB15K237 	|               3               	|  	|  	|  	|  	|  	|  	|
| FB15K237 	|                    4                    	|  	|  	|  	|  	|  	|  	|


#### Results
---


Faire 2 versions du tableau 1,


Tableau distillation d'entités puis distillation de relations et both.


Compléter tableau 2.

Expliquer qu'on utilise un sous ensemble (sampling) et qu'on inclut systématiquement la vérité terrain. Faiblesse du modèle. On pourrait sélectionner le top k des entités.

Dans la verticale du tableau 1, on analyse l'impacte de la base de connaissance dans la qualité de la représentation et dans l'horyzontal, nous validons une stratégie de distillation.

Quesque j'aurai fait si deux personnes ont deux bases de connaissances. 

Xdistill~X distillation séquentielle 
Our model distillation coopérative

2 propositions


18.29 == 18.13

Finir une itération d'expérience



