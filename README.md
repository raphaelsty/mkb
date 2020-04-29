#### KD-MKDR


Installation:
```
pip install -r requirements.txt
python setup.py install
```


| experiment | target |  realization  | results | discussion   | perspectives |
|   :---:    | :---:  |   :---:       | :---:   |     :---:    | :---:       |
||||||
| Join fb15k237 with Wordnet18rr. | Improving RotatE results on the dataset fb15k237 | To identify common entities between the two KGs, I find triplets that have a common head and tail between the two KG. I make the hypothesize that the triplets that have a common head and tail in both KG contain common entities. I identify 464 triplets with identical head and tail that have the relation ``_has_part`` in wn18rr and ``location/location/contains`` in fb15k237. I therefore choose to replace the relation "location/location/contains" in fb15k237 by the relation "_has_part" in wn18rr. These are the only two relationships I can bring together. | I built a dataset common to wn18rr and fb15k237 containing 638 entities and a single relationship ``_has_part``. This dataset is composed of 1234 triplets. It contains exclusively geographical information. 56.08% of triplets comes from fb15k237 and 43.92% comes from wn18rr. | The granularity of the relations of fb15k237 is finer than the one of wn18rr. This experiment does not allow to establish many links between the two KGs. 20.77% of the triplets in the validation set of fb15k237 contain at least one entity common to both KGs.  Only 0.7% of the triplets in the fb15K237 validation set contain two entities common to both KGs. 20.96% of the triplets in the fb15k237 test set contain at least one entity common to both KGs. Only 0.6% of the triplets in the fb15K237 test set contain two entities common to both KGs. 3.89% of the triplets in the wn18rr validation set contain at least one entity common to both KGs.  Only 0.5% of the triplets in the wn18rr validation set contain two entities common to both KGs. 3.98% of the triplets in the wn18rr test set contain at least one entity common to both KGs. Only 0.6% of the triplets in the wn18rr test set contain two entities common to both KGs.| I think the dataset I created may have bigger impact on fb15k237. Are my models already trained on wn18rr / fb15k237 experts on this new dataset? How much can be gained?|
|||||||
|||||||
|||||||
|||||||
|||||||

