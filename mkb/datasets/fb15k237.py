import pathlib

from ..utils import read_csv, read_csv_classification, read_json
from .dataset import Dataset

__all__ = ["Fb15k237"]


class Fb15k237(Dataset):
    """Fb15k237 dataset.

    Fb15k237 aim to iterate over the associated dataset. It provide positive samples, corresponding
    weights and the mode (head batch / tail batch)

    Parameters:
        batch_size (int): Size of the batch.
        classification (bool): Must be set to True when using ConvE model to optimize BCELoss.
        shuffle (bool): Whether to shuffle the dataset or not.
        pre_compute (bool): Pre-compute parameters such as weights when using translationnal model
            (TransE, DistMult, RotatE, pRotatE, ComplEx). When using ConvE, pre-compute target
            matrix. When pre_compute is set to True, the model training is faster but it needs more
            memory.
        num_workers (int): Number of workers dedicated to iterate on the dataset.
        seed (int): Random state.

    Attributes:
        train (list): Training set.
        valid (list): Validation set.
        test (list): Testing set.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.

    Example:

        >>> from mkb import datasets

        >>> dataset = datasets.Fb15k237(batch_size=1, shuffle=False, pre_compute=False, seed=42)

        >>> dataset
        Fb15k237 dataset
            Batch size  1
            Entities  14541
            Relations  237
            Shuffle  False
            Train triples  272115
            Validation triples  17535
            Test triples  20466

        >>> assert len(dataset.classification_valid['X']) == len(dataset.classification_valid['y'])
        >>> assert len(dataset.classification_test['X']) == len(dataset.classification_test['y'])

        >>> assert len(dataset.classification_valid['X']) == len(dataset.valid) * 2
        >>> assert len(dataset.classification_test['X']) == len(dataset.test) * 2

    References:
        1. [Toutanova, Kristina, et al. "Representing text for joint embedding of text and knowledge bases." Proceedings of the 2015 conference on empirical methods in natural language processing. 2015.](https://www.aclweb.org/anthology/D15-1174.pdf)
        2. [Datasets for Knowledge Graph Completion with Textual Information about Entities](https://github.com/villmow/datasets_knowledge_embedding)

    """

    def __init__(
        self,
        batch_size,
        classification=False,
        shuffle=True,
        pre_compute=True,
        num_workers=1,
        seed=None,
    ):

        self.filename = "fb15k237"

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(file_path=f"{path}/train.csv"),
            valid=read_csv(file_path=f"{path}/valid.csv"),
            test=read_csv(file_path=f"{path}/test.csv"),
            entities=read_json(f"{path}/entities.json"),
            relations=read_json(f"{path}/relations.json"),
            batch_size=batch_size,
            shuffle=shuffle,
            classification=classification,
            pre_compute=pre_compute,
            num_workers=num_workers,
            seed=seed,
            classification_valid=read_csv_classification(f"{path}/classification_valid.csv"),
            classification_test=read_csv_classification(f"{path}/classification_test.csv"),
        )
